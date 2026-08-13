#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SmoothQuant 迁移强度 α 扫描(W8A8 伪量化口径)
=============================================

SmoothQuant:激活难量化、权重好量化 → 把激活的难度**迁移**一部分给权重:
  Y = (X/s) · (s·W),  s_j = max|X_j|^α / max|W_j|^(1-α)
  α=0 难度全留激活,α=1 全推给权重,论文经验 0.5~0.8。

与 AWQ 保护指数 α 的区别:那边只量化权重(W4A16),这边权重和激活都量化
(W8A8),s 的分子分母同时看两边的 absmax。

模拟:目标 llm down_proj 组(激活离群值 1.59e6 宿主,最严苛考场)。
  权重:W8 = int8 per-out-row 伪量化(s·W)
  激活:A8 = int8 per-token 动态伪量化(X/s),用 forward_pre_hook 注入
基线:s=1(不迁移的裸 W8A8)。量输出 KL(对 FP16)。
"""

import argparse, json
from pathlib import Path
import torch
import numpy as np


def fwd(model, s, device):
    return model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                 attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                 pixel_values=s["pixel_values"].to(device, torch.float16),
                 image_grid_thw=s["image_grid_thw"].to(device))


def collect_absmax(model, samples, device, targets, n_stat=16):
    acc, hooks = {}, []
    mods = dict(model.named_modules())

    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().abs()
            m = x.amax(0)
            acc[name] = torch.maximum(acc.get(name, torch.zeros_like(m)), m)
        return hook

    for n in targets:
        hooks.append(mods[n].register_forward_hook(mk(n)))
    with torch.no_grad():
        for s in samples[:n_stat]:
            fwd(model, s, device)
    for h in hooks: h.remove()
    return acc


def q8_perrow(w32):
    sc = w32.abs().amax(-1, keepdim=True) / 127.0
    return torch.round(w32 / (sc + 1e-12)).clamp(-127, 127) * sc


def q8_pertoken(x32):
    sc = x32.abs().amax(-1, keepdim=True) / 127.0
    return torch.round(x32 / (sc + 1e-12)).clamp(-127, 127) * sc


def probes_logits(model, samples, device, n=8):
    outs = []
    with torch.no_grad():
        for s in samples[:n]:
            outs.append(fwd(model, s, device).logits[0, -1].float().cpu())
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\models\qwen25vl-3b")
    ap.add_argument("--calib", default="calib/data/vl_calib_128.pt")
    ap.add_argument("--out", default="results/raw/sq_alpha_scan_desktop.json")
    args = ap.parse_args()
    device = "cuda"

    from transformers import AutoModelForVision2Seq
    print("[1/4] 加载模型与校准样本")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    samples = torch.load(args.calib, weights_only=False)["samples"]

    mods = dict(model.named_modules())
    targets = [n for n, m in mods.items() if isinstance(m, torch.nn.Linear)
               and "down_proj" in n and not any(k in n for k in ("visual", "vision", "merger"))]
    print(f"    目标:llm down_proj × {len(targets)}(W8A8 模拟)")

    print("[2/4] 统计激活/权重 absmax")
    x_max = collect_absmax(model, samples, device, targets)
    w_max = {n: mods[n].weight.data.float().abs().amax(0) for n in targets}  # 每输入通道

    print("[3/4] FP16 基线")
    baseline = probes_logits(model, samples, device)
    base_p = [b.softmax(-1) for b in baseline]

    saved = {n: mods[n].weight.data.clone() for n in targets}
    configs = [("s=1(裸W8A8)", None), ("α=0.3", 0.3), ("α=0.5", 0.5),
               ("α=0.7", 0.7), ("α=0.9", 0.9)]
    curve = []
    print("[4/4] 扫描")
    for label, a in configs:
        s_map, hooks = {}, []
        for n in targets:
            if a is None:
                s = torch.ones_like(w_max[n])
            else:
                s = (x_max[n].to(device) + 1e-8).pow(a) / (w_max[n].to(device) + 1e-8).pow(1 - a)
                s = s.clamp(min=1e-5)
            s_map[n] = s
            mods[n].weight.data = q8_perrow(saved[n].float() * s.unsqueeze(0)).to(torch.float16)

        def mk_pre(name):
            def pre(mod, args_in):
                x = args_in[0]
                xq = q8_pertoken(x.float() / s_map[name]).to(x.dtype)
                return (xq,) + tuple(args_in[1:])
            return pre
        for n in targets:
            hooks.append(mods[n].register_forward_pre_hook(mk_pre(n)))

        probes = probes_logits(model, samples, device)
        for h in hooks: h.remove()
        for n in targets: mods[n].weight.data = saved[n]

        kls = [float((bp * (bp / (q.softmax(-1) + 1e-9)).log()).sum())
               for bp, q in zip(base_p, probes)]
        rec = {"config": label, "alpha": a, "kl_mean": float(np.mean(kls))}
        curve.append(rec)
        print(f"  {label:12s} KL={rec['kl_mean']:.6f}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "target_group": "llm_mlp_down(36)",
        "quant": "W8A8 sim: weight int8 per-row static, act int8 per-token dynamic",
        "note": "down_proj 输入非 LayerNorm 直连,部署时 1/s 融合受限——本实验只看迁移机制",
        "curve": curve,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    main()
