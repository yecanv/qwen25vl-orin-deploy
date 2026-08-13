#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AWQ α 扫描实验
==============

问题:AWQ 的保护公式 s_j = mean(|x_j|)^α 里,α 从 0 到 1 扫过去,
量化误差应该走出一条 U 型曲线:
  α=0   完全不保护 → 等价于朴素 RTN,大激活通道的权重误差被放大
  α 适中 保护 salient 通道,同组代价可接受 → 误差最低点
  α=1   过度保护 → 非 salient 通道被抬高的 scale 连累,误差回升

方法(与 sensitivity_analysis.py 同一框架,三处升级):
  1. 统计目标模块每个输入通道的 mean(|x|)   ← AWQ 用均值,敏感度分析用 max
  2. 等价变换 W·diag(s) 后做 INT4 group-128 伪量化,再 ·diag(1/s) 还原
  3. 用输出 logits 的 KL 散度(对 FP16 基线)+ MSE 度量损伤

目标模块组:llm 全部 down_proj(36 层)——敏感度分析实测的最难组
(激活离群值 1.59e6,INT8 伪量化 KL 最大)。
"""

import argparse, json
from pathlib import Path

import torch
import numpy as np


def collect_mean_abs(model, samples, device, targets, n_stat=16):
    """逐输入通道统计 mean(|x|)——AWQ 的'体检'步骤"""
    acc = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            x = x.reshape(-1, x.shape[-1]).float().abs()
            rec = acc.setdefault(name, {"sum": torch.zeros(x.shape[-1], device=x.device),
                                        "n": 0})
            rec["sum"] += x.sum(dim=0)
            rec["n"] += x.shape[0]
        return hook

    mods = dict(model.named_modules())
    for name in targets:
        hooks.append(mods[name].register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for s in samples[:n_stat]:
            model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                  attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                  pixel_values=s["pixel_values"].to(device, torch.float16),
                  image_grid_thw=s["image_grid_thw"].to(device))
    for h in hooks:
        h.remove()
    return {k: (v["sum"] / v["n"]) for k, v in acc.items()}


def int4_g128_fake_quant(w32: torch.Tensor, group=128):
    """对 [out,in] 权重沿输入维做 INT4 对称分组伪量化(格子 -7..7 用满对称档)"""
    out_f, in_f = w32.shape
    assert in_f % group == 0
    wg = w32.reshape(out_f, in_f // group, group)
    scale = wg.abs().amax(dim=-1, keepdim=True) / 7.0
    q = torch.round(wg / (scale + 1e-12)).clamp(-7, 7)
    return (q * scale).reshape(out_f, in_f)


def forward_probes(model, samples, device, n_probe):
    outs = []
    with torch.no_grad():
        for s in samples[:n_probe]:
            o = model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                      attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                      pixel_values=s["pixel_values"].to(device, torch.float16),
                      image_grid_thw=s["image_grid_thw"].to(device))
            outs.append(o.logits[0, -1].float().cpu())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\models\qwen25vl-3b")
    ap.add_argument("--calib", default="calib/data/vl_calib_128.pt")
    ap.add_argument("--out", default="results/raw/alpha_scan_desktop.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--n-probe", type=int, default=8)
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq

    print("[1/4] 加载模型与校准样本")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True)
    pack = torch.load(args.calib, weights_only=False)
    samples = pack["samples"]

    mods = dict(model.named_modules())
    targets = [n for n, m in mods.items()
               if isinstance(m, torch.nn.Linear)
               and "down_proj" in n
               and not any(k in n for k in ("visual", "vision", "merger"))]
    print(f"    目标模块:llm down_proj × {len(targets)}")

    print("[2/4] 体检:统计逐通道 mean(|x|)")
    mean_abs = collect_mean_abs(model, samples, args.device, targets)

    print("[3/4] FP16 基线前向")
    baseline = forward_probes(model, samples, args.device, args.n_probe)
    base_p = [b.softmax(-1) for b in baseline]

    saved = {n: mods[n].weight.data.clone() for n in targets}
    alphas = [float(a) for a in args.alphas.split(",")]
    curve = []

    print("[4/4] α 扫描")
    for a in alphas:
        for n in targets:
            w32 = saved[n].float()
            s = (mean_abs[n].to(w32.device) + 1e-8).pow(a)      # s_j = mean|x_j|^α
            wq = int4_g128_fake_quant(w32 * s.unsqueeze(0))      # 变换后量化
            mods[n].weight.data = (wq / s.unsqueeze(0)).to(torch.float16)  # 除回

        probes = forward_probes(model, samples, args.device, args.n_probe)
        kls, mses = [], []
        for bp, b, q in zip(base_p, baseline, probes):
            qp = q.softmax(-1)
            kls.append(float((bp * (bp / (qp + 1e-9)).log()).sum()))
            mses.append(float(((b - q) ** 2).mean()))
        rec = {"alpha": a,
               "kl_mean": float(np.mean(kls)),
               "mse_mean": float(np.mean(mses))}
        curve.append(rec)
        print(f"  α={a:<4}  KL={rec['kl_mean']:.6f}  logitsMSE={rec['mse_mean']:.6f}")

    for n in targets:                                            # 还原
        mods[n].weight.data = saved[n]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "calib": args.calib,
        "target_group": "llm_mlp_down(36)",
        "quant": "int4 symmetric group-128, equivalent transform w*s, s=mean|x|^alpha",
        "n_probe": args.n_probe,
        "curve": curve,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [c["alpha"] for c in curve]
        fig, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(xs, [c["kl_mean"] for c in curve], "o-", label="KL vs FP16")
        ax1.set_xlabel("alpha")
        ax1.set_ylabel("KL divergence")
        ax1.set_title("AWQ alpha scan: llm_mlp_down, INT4-g128")
        ax1.grid(alpha=0.3)
        png = out.with_suffix(".png")
        fig.tight_layout(); fig.savefig(png, dpi=140)
        print(f"→ {png}")
    except Exception as e:
        print("绘图跳过:", e)


if __name__ == "__main__":
    main()
