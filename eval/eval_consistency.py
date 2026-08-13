#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
量化前后精度回归
================
三个层次，从粗到细：

1. 逐 token 一致性  —— 贪心解码下，量化前后输出多少 token 后开始分叉
2. 任务级准确率     —— MMBench / TextVQA 上的实际得分
3. 分布距离         —— 首 token logits 的 KL 散度

第 1 项最直观："量化后前 N 个 token 完全一致，第 N+1 个开始分叉"，
比只报一个准确率有说服力得多。

红线：评测集与校准集严格分离。校准用各数据集的 train/val 采样，
评测用 test split。同一批数据就是泄漏。
"""
import argparse, json
from pathlib import Path
from typing import List, Dict

import numpy as np


def first_divergence(a: List[int], b: List[int]) -> int:
    """返回两个 token 序列首次不同的位置；完全相同返回 len。"""
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def evaluate(pipe_fp16, pipe_quant, samples: List[Dict],
             max_new_tokens: int = 128) -> Dict:
    divergences, exact_match = [], 0
    fp16_lens, quant_lens = [], []

    for i, s in enumerate(samples):
        r_a = pipe_fp16.generate(s["image"], s["prompt"],
                                 max_new_tokens=max_new_tokens, stream=False)
        r_b = pipe_quant.generate(s["image"], s["prompt"],
                                  max_new_tokens=max_new_tokens, stream=False)

        ta = pipe_fp16.tok(r_a["text"])["input_ids"]
        tb = pipe_quant.tok(r_b["text"])["input_ids"]

        d = first_divergence(ta, tb)
        divergences.append(d)
        fp16_lens.append(len(ta))
        quant_lens.append(len(tb))
        if r_a["text"].strip() == r_b["text"].strip():
            exact_match += 1

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(samples)}  "
                  f"平均分叉位置 {np.mean(divergences):.1f}")

    div = np.array(divergences)
    return {
        "n_samples": len(samples),
        "exact_match_rate": round(exact_match / len(samples), 4),
        "divergence": {
            "mean": round(float(div.mean()), 2),
            "p50": int(np.median(div)),
            "p10": int(np.percentile(div, 10)),
            "never_diverged_rate": round(float((div >= np.array(fp16_lens)).mean()), 4),
        },
        "length": {
            "fp16_mean": round(float(np.mean(fp16_lens)), 1),
            "quant_mean": round(float(np.mean(quant_lens)), 1),
        },
        "解读": "分叉位置越靠后，量化对生成的影响越小。"
                "若 p10 分叉位置 < 5，说明量化明显改变了模型行为，需要检查。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vit-engine", default="engines/vit_fp16.engine")
    ap.add_argument("--llm-fp16", default="engines/llm_fp16")
    ap.add_argument("--llm-quant", default="engines/llm_int4awq")
    ap.add_argument("--eval-set", default="eval/data/eval_100.json",
                    help="与校准集无交集的评测样本")
    ap.add_argument("--out", default="results/raw/consistency.json")
    args = ap.parse_args()

    from runtime.run_vl import VLPipeline

    samples = json.loads(Path(args.eval_set).read_text(encoding="utf-8"))
    print(f"评测样本 {len(samples)} 条")

    print("加载 FP16 基线…")
    p_fp16 = VLPipeline(args.model, args.vit_engine, args.llm_fp16)
    print("加载量化版…")
    p_q = VLPipeline(args.model, args.vit_engine, args.llm_quant)

    r = evaluate(p_fp16, p_q, samples)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(r, ensure_ascii=False, indent=2),
                              encoding="utf-8")
    print(json.dumps(r, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
