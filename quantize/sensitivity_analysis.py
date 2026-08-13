#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
逐模块量化敏感度分析
====================

这个脚本是"为什么 ViT 不量化"这个决策的**证据来源**。

面试官问「为什么视觉编码器保 FP16」，两种答法：
  差：「业界都这么做 / ViT 对量化敏感」          → 背的
  好：「我做了逐模块敏感度扫描，ViT 的 patch_embed
       和 merger 量化后 PPL 涨了 X，OCR 准确率掉了 Y，
       而 LLM 主干的 MLP 层只掉 Z」                → 跑过的

方法
----
逐个模块单独量化（其余保持 FP16），测量：
  1. 输出 logits 与全 FP16 基线的 KL 散度
  2. 该模块激活值的 outlier 强度（max/median 比值）
  3. 该模块的参数量占比（衡量量化收益）

画成散点图：X 轴 = 参数占比（收益），Y 轴 = KL 散度（代价）。
右下角的模块该量化，左上角的不该量化。这张图是项目里最有说服力的一张。
"""

import argparse, json
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np


def collect_activation_stats(model, samples, device, max_batch=8) -> Dict[str, Dict]:
    """
    统计每个 Linear 层输入激活的 outlier 强度。

    outlier_ratio = per-channel absmax 的 max / median
    这个值越大，说明激活里有少数 channel 数值远超其他，
    per-tensor 量化会被这几个 channel 拉爆 scale。

    经验阈值：
      < 10   量化友好
      10-50  需要 SmoothQuant 做迁移
      > 50   建议保 FP16
    """
    stats: Dict[str, Dict] = {}
    hooks = []

    def make_hook(name):
        def hook(mod, inp, out):
            x = inp[0].detach()
            if x.dim() < 2:
                return
            x = x.reshape(-1, x.shape[-1]).float().abs()
            chan_max = x.amax(dim=0)                    # [hidden]
            rec = stats.setdefault(name, {
                "chan_absmax": torch.zeros_like(chan_max),
                "n": 0,
                "params": sum(p.numel() for p in mod.parameters()),
            })
            rec["chan_absmax"] = torch.maximum(rec["chan_absmax"], chan_max)
            rec["n"] += 1
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        for s in samples[:max_batch]:
            try:
                model(
                    input_ids=s["input_ids"].unsqueeze(0).to(device),
                    attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                    pixel_values=s["pixel_values"].to(device, dtype=torch.float16),
                    image_grid_thw=s["image_grid_thw"].to(device),
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                continue

    for h in hooks:
        h.remove()

    out = {}
    for name, rec in stats.items():
        a = rec["chan_absmax"]
        med = a.median().item()
        mx = a.max().item()
        out[name] = {
            "outlier_ratio": round(mx / (med + 1e-9), 2),
            "absmax": round(mx, 4),
            "median": round(med, 6),
            "params": rec["params"],
            "is_vision": any(k in name for k in
                             ("visual", "vision", "merger", "patch_embed")),
        }
    return out


def kl_divergence_probe(model, samples, device, target_modules: List[str],
                        n_probe: int = 8) -> float:
    """
    伪量化指定模块（round-to-nearest INT8），测输出 logits 与 FP16 基线的 KL。
    这是快速筛选，不如真跑一遍 ModelOpt 准，但能在几分钟内扫完所有模块。
    """
    def fake_quant_(t: torch.Tensor, bits: int = 8):
        s = t.abs().amax(dim=-1, keepdim=True) / (2 ** (bits - 1) - 1)
        return torch.round(t / (s + 1e-9)).clamp(-(2 ** (bits - 1)),
                                                 2 ** (bits - 1) - 1) * s

    baseline, quantized = [], []
    saved = {}

    with torch.no_grad():
        # 基线
        for s in samples[:n_probe]:
            o = model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                      attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                      pixel_values=s["pixel_values"].to(device, torch.float16),
                      image_grid_thw=s["image_grid_thw"].to(device))
            baseline.append(o.logits[0, -1].float().softmax(-1).cpu())

        # 伪量化目标模块
        for name, mod in model.named_modules():
            if name in target_modules and hasattr(mod, "weight"):
                saved[name] = mod.weight.data.clone()
                mod.weight.data = fake_quant_(mod.weight.data)

        for s in samples[:n_probe]:
            o = model(input_ids=s["input_ids"].unsqueeze(0).to(device),
                      attention_mask=s["attention_mask"].unsqueeze(0).to(device),
                      pixel_values=s["pixel_values"].to(device, torch.float16),
                      image_grid_thw=s["image_grid_thw"].to(device))
            quantized.append(o.logits[0, -1].float().softmax(-1).cpu())

        # 还原
        for name, mod in model.named_modules():
            if name in saved:
                mod.weight.data = saved[name]

    kls = [float((p * (p / (q + 1e-9)).log()).sum())
           for p, q in zip(baseline, quantized)]
    return float(np.mean(kls))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--calib", default="calib/data/vl_calib_512.pt")
    ap.add_argument("--out", default="results/raw/sensitivity.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-kl", action="store_true", help="只跑激活统计，快很多")
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq

    print("[1/3] 加载模型与校准样本")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True)
    pack = torch.load(args.calib, weights_only=False)
    samples = pack["samples"]

    print("[2/3] 统计逐层激活 outlier")
    stats = collect_activation_stats(model, samples, args.device)

    print("[3/3] 分组 KL 探测")
    groups = {
        "vision_patch_embed": [n for n in stats if "patch_embed" in n],
        "vision_attn":        [n for n in stats if stats[n]["is_vision"] and "attn" in n],
        "vision_mlp":         [n for n in stats if stats[n]["is_vision"] and "mlp" in n],
        "vision_merger":      [n for n in stats if "merger" in n],
        "llm_attn_qkv":       [n for n in stats if not stats[n]["is_vision"]
                               and any(k in n for k in ("q_proj", "k_proj", "v_proj"))],
        "llm_attn_o":         [n for n in stats if not stats[n]["is_vision"] and "o_proj" in n],
        "llm_mlp_gate_up":    [n for n in stats if not stats[n]["is_vision"]
                               and any(k in n for k in ("gate_proj", "up_proj"))],
        "llm_mlp_down":       [n for n in stats if not stats[n]["is_vision"] and "down_proj" in n],
        "lm_head":            [n for n in stats if "lm_head" in n],
    }

    report = []
    total_params = sum(v["params"] for v in stats.values())
    for g, mods in groups.items():
        if not mods:
            continue
        params = sum(stats[m]["params"] for m in mods)
        ratios = [stats[m]["outlier_ratio"] for m in mods]
        kl = None if args.skip_kl else kl_divergence_probe(
            model, samples, args.device, mods)
        report.append({
            "group": g,
            "n_modules": len(mods),
            "params": params,
            "param_share_pct": round(params / total_params * 100, 2),
            "outlier_ratio_max": round(max(ratios), 2),
            "outlier_ratio_p50": round(float(np.median(ratios)), 2),
            "kl_int8": round(kl, 6) if kl is not None else None,
        })
        print(f"  {g:22s} 参数占比 {report[-1]['param_share_pct']:5.2f}%  "
              f"outlier(max) {report[-1]['outlier_ratio_max']:8.2f}  "
              f"KL {report[-1]['kl_int8']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model,
        "calib": args.calib,
        "groups": report,
        "per_module": stats,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n→ {out}")
    print("下一步：python benchmark/plot_results.py --sensitivity  出散点图")
    print("结论写进 docs/03，面试被问'为什么 ViT 不量化'就拿这张图。")


if __name__ == "__main__":
    main()
