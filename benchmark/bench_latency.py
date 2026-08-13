#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
延迟分解 Benchmark
==================
VLM 的 TTFT 和纯语言模型不是一回事，必须拆开报：

    端到端 TTFT = 图像预处理 + ViT 编码 + prompt 拼接 + LLM prefill

只报一个总数是没有说服力的。拆开之后你才能回答面试官的追问：
"你这 TTFT 里视觉编码占多少？"——这是判断优化方向的依据。

同时扫描不同分辨率，因为视觉 token 数是 TTFT 的主导因素。
"""
import argparse, json, statistics, time
from pathlib import Path


RESOLUTIONS = [
    ("低 (~256 tok)",  200704),
    ("中 (~576 tok)",  451584),
    ("高 (~1024 tok)", 802816),
    ("超高(~2048 tok)",1605632),
]


def bench_one(pipe, image, prompt, max_pixels, n_warmup=3, n_iter=20):
    pipe.proc.image_processor.max_pixels = max_pixels
    for _ in range(n_warmup):
        pipe.generate(image, prompt, max_new_tokens=32, stream=False)

    recs = []
    for _ in range(n_iter):
        r = pipe.generate(image, prompt, max_new_tokens=128, stream=False)
        recs.append(r)

    def agg(key):
        xs = [r[key] for r in recs]
        return {
            "mean": round(statistics.mean(xs), 2),
            "p50": round(statistics.median(xs), 2),
            "p95": round(sorted(xs)[int(len(xs) * 0.95)], 2),
            "std": round(statistics.stdev(xs), 2) if len(xs) > 1 else 0.0,
        }

    return {
        "max_pixels": max_pixels,
        "n_visual_tokens": recs[0]["n_visual_tokens"],
        "n_prompt_tokens": recs[0]["n_prompt_tokens"],
        "vit_ms": agg("vit_ms"),
        "llm_prefill_ms": agg("llm_ttft_ms"),
        "e2e_ttft_ms": agg("ttft_ms"),
        "decode_tok_s": agg("decode_tok_s"),
        "vit_share_pct": round(
            statistics.mean([r["vit_ms"] / r["ttft_ms"] * 100 for r in recs]), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vit-engine", default="engines/vit_fp16.engine")
    ap.add_argument("--llm-engine", default="engines/llm_int4awq")
    ap.add_argument("--image", default="assets/demo.jpg")
    ap.add_argument("--prompt", default="描述这张图片。")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--tag", default="int4awq", help="写进结果文件，用于方案对比")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from runtime.run_vl import VLPipeline
    pipe = VLPipeline(args.model, args.vit_engine, args.llm_engine)

    results = []
    for name, mp in RESOLUTIONS:
        print(f"\n=== {name}  max_pixels={mp} ===")
        try:
            r = bench_one(pipe, args.image, args.prompt, mp, n_iter=args.iters)
            r["resolution_label"] = name
            results.append(r)
            print(f"  视觉 token {r['n_visual_tokens']:5d}  "
                  f"ViT {r['vit_ms']['p50']:7.2f}ms ({r['vit_share_pct']}% of TTFT)  "
                  f"e2e TTFT {r['e2e_ttft_ms']['p50']:7.2f}ms  "
                  f"decode {r['decode_tok_s']['p50']:.1f} tok/s")
        except Exception as e:
            print(f"  失败: {e}")
            print(f"  若为 shape 错误，检查 build_vit_engine.sh 的 MAX_PATCHES "
                  f"是否覆盖 {mp // 784 * 4}")

    out = Path(args.out or f"results/raw/latency_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": args.tag,
        "cold_start_s": round(pipe.cold_start_s, 2),
        "device": json.loads(Path("results/device_info.json").read_text())
                  if Path("results/device_info.json").exists() else None,
        "results": results,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
