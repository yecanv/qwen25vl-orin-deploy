#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图文校准 vs 纯文本校准 对照实验(伪量化口径)
=============================================

问题:VLM(视觉语言模型)量化时,校准集(calibration set,统计激活用的样本)
必须是图文混合的吗?纯文本校准会伤什么?

方法(2×2 四象限):
  校准来源 ∈ {图文混合, 纯文本} × 探针任务 ∈ {图文, 纯文本}
  对 llm down_proj 组做 AWQ 式保护(k=mean|x|^0.2,α 取此前实测最优)
  + INT4 group-128 伪量化,量输出 KL(对各自 FP16 基线)。

预期:纯文本校准的 k 没见过视觉 token 的激活分布,
     在图文探针上应显著劣于图文校准;纯文本探针两者应接近。
"""

import argparse, json
from pathlib import Path
import torch
import numpy as np

TEXT_PROMPTS = [
    "请解释什么是操作系统的虚拟内存,以及页表在其中的作用。",
    "写一段快速排序的伪代码,并说明平均时间复杂度。",
    "总结一下光合作用的基本过程。",
    "What are the main differences between TCP and UDP?",
    "请把下面这句话翻译成英文:今天天气很好,我们去公园散步吧。",
    "解释一下什么是通货膨胀,它对普通人的生活有什么影响?",
    "Describe the water cycle in nature.",
    "列举三种常见的排序算法并比较它们的稳定性。",
    "什么是嵌入式系统?请举两个生活中的例子。",
    "Explain the concept of recursion with a simple example.",
    "请写一首关于秋天的五言绝句。",
    "数据库中的事务具有哪四个特性?分别解释。",
    "What is the difference between a compiler and an interpreter?",
    "简述牛顿三大运动定律。",
    "请解释 HTTP 和 HTTPS 的区别。",
    "描述一下如何煮一碗好吃的面条。",
    "什么是机器学习中的过拟合?如何缓解?",
    "Explain why the sky appears blue during the day.",
    "请比较数组和链表的优缺点。",
    "介绍一下中国的春节习俗。",
    "What are the advantages of solid state drives over hard disk drives?",
    "解释一下什么是复利,并举一个例子。",
    "简述计算机启动(开机)过程中发生了什么。",
    "请给一位刚学编程的朋友三条建议。",
]


def build_text_samples(tokenizer, prompts, device):
    out = []
    for p in prompts:
        msgs = [{"role": "user", "content": p}]
        ids = tokenizer.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt")[0]
        out.append({"input_ids": ids,
                    "attention_mask": torch.ones_like(ids)})
    return out


def fwd(model, s, device):
    kw = dict(input_ids=s["input_ids"].unsqueeze(0).to(device),
              attention_mask=s["attention_mask"].unsqueeze(0).to(device))
    if "pixel_values" in s:
        kw["pixel_values"] = s["pixel_values"].to(device, torch.float16)
        kw["image_grid_thw"] = s["image_grid_thw"].to(device)
    return model(**kw)


def collect_mean_abs(model, samples, device, targets, n_stat=16):
    acc, hooks = {}, []
    mods = dict(model.named_modules())

    def mk(name):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float().abs()
            rec = acc.setdefault(name, {"sum": torch.zeros(x.shape[-1], device=x.device), "n": 0})
            rec["sum"] += x.sum(0); rec["n"] += x.shape[0]
        return hook

    for n in targets:
        hooks.append(mods[n].register_forward_hook(mk(n)))
    with torch.no_grad():
        for s in samples[:n_stat]:
            fwd(model, s, device)
    for h in hooks: h.remove()
    return {k: v["sum"] / v["n"] for k, v in acc.items()}


def int4_g128(w32, group=128):
    o, i = w32.shape
    wg = w32.reshape(o, i // group, group)
    sc = wg.abs().amax(-1, keepdim=True) / 7.0
    return (torch.round(wg / (sc + 1e-12)).clamp(-7, 7) * sc).reshape(o, i)


def probes_logits(model, samples, device, n=8):
    outs = []
    with torch.no_grad():
        for s in samples[:n]:
            outs.append(fwd(model, s, device).logits[0, -1].float().cpu())
    return outs


def kl(base, quant):
    vals = []
    for b, q in zip(base, quant):
        bp, qp = b.softmax(-1), q.softmax(-1)
        vals.append(float((bp * (bp / (qp + 1e-9)).log()).sum()))
    return float(np.mean(vals))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"E:\models\qwen25vl-3b")
    ap.add_argument("--calib", default="calib/data/vl_calib_128.pt")
    ap.add_argument("--out", default="results/raw/calib_ablation_desktop.json")
    ap.add_argument("--alpha", type=float, default=0.2)
    args = ap.parse_args()
    device = "cuda"

    from transformers import AutoModelForVision2Seq, AutoTokenizer
    print("[1/5] 加载模型/分词器/校准包")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map=device, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vl_samples = torch.load(args.calib, weights_only=False)["samples"]
    txt_samples = build_text_samples(tok, TEXT_PROMPTS, device)

    mods = dict(model.named_modules())
    targets = [n for n, m in mods.items() if isinstance(m, torch.nn.Linear)
               and "down_proj" in n and not any(k in n for k in ("visual", "vision", "merger"))]
    print(f"    目标:llm down_proj × {len(targets)},α={args.alpha}")

    print("[2/5] 两种校准来源各自统计 mean|x|")
    stats = {"vl": collect_mean_abs(model, vl_samples, device, targets),
             "txt": collect_mean_abs(model, txt_samples, device, targets)}

    print("[3/5] 双探针 FP16 基线")
    probe = {"vl": vl_samples[16:24], "txt": txt_samples[16:24]}
    base = {p: probes_logits(model, probe[p], device) for p in probe}

    saved = {n: mods[n].weight.data.clone() for n in targets}
    result = {}
    print("[4/5] 四象限扫描")
    for calib in ("vl", "txt"):
        for n in targets:
            w32 = saved[n].float()
            k = (stats[calib][n].to(w32.device) + 1e-8).pow(args.alpha)
            mods[n].weight.data = (int4_g128(w32 * k.unsqueeze(0)) / k.unsqueeze(0)).to(torch.float16)
        for p in ("vl", "txt"):
            result[f"calib={calib}|probe={p}"] = kl(base[p], probes_logits(model, probe[p], device))
            print(f"  calib={calib:3s} probe={p:3s}  KL={result[f'calib={calib}|probe={p}']:.6f}")
        for n in targets:
            mods[n].weight.data = saved[n]

    print("[5/5] 落盘")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "model": args.model, "alpha": args.alpha,
        "target_group": "llm_mlp_down(36)",
        "quant": "int4-g128 + AWQ-style k=mean|x|^alpha",
        "quadrants": result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    main()
