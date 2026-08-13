#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
融合算子 benchmark
==================
两层测量，面试要分开讲：

  第一层：kernel 级 —— 融合 vs 朴素三段式
          省的是访存和 launch，不是算力

  第二层：端到端 —— token 压缩后 LLM prefill 的收益
          attention O(n²)，这才是大头

所有数字必须在你自己的 Orin 上跑出来。
"""
import argparse, json, time
from pathlib import Path


def bench_kernel_level(N, D, iters=100, warmup=20):
    import torch
    import token_merge_cuda as tm

    tokens = torch.randn(N, D, device="cuda", dtype=torch.float16)

    # ---- 融合版 ----
    for _ in range(warmup):
        tm.fused_match(tokens)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        idx, val = tm.fused_match(tokens)
    torch.cuda.synchronize()
    fused_ms = (time.perf_counter() - t0) / iters * 1000

    # ---- 朴素三段式（用 PyTorch 算子模拟：独立归一化 → 物化 S 矩阵 → 逐行 argmax）----
    def naive():
        a = tokens[0::2].float()
        b = tokens[1::2].float()
        a = a / (a.norm(dim=-1, keepdim=True) + 1e-6)     # kernel ①
        b = b / (b.norm(dim=-1, keepdim=True) + 1e-6)
        S = a @ b.T                                        # kernel ② 物化 S
        return S.max(dim=-1)                               # kernel ③

    for _ in range(warmup):
        naive()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        naive()
    torch.cuda.synchronize()
    naive_ms = (time.perf_counter() - t0) / iters * 1000

    # ---- 正确性交叉验证 ----
    idx_f, val_f = tm.fused_match(tokens)
    val_n, idx_n = naive()
    match = (idx_f.cpu() == idx_n.cpu().int()).float().mean().item()
    val_diff = (val_f.cpu() - val_n.cpu()).abs().max().item()

    Na, Nb = (N + 1) // 2, N // 2
    return {
        "n_tokens": N, "dim": D,
        "fused_ms": round(fused_ms, 4),
        "naive_ms": round(naive_ms, 4),
        "speedup": round(naive_ms / fused_ms, 3),
        "S_matrix_MB": round(Na * Nb * 4 / 1e6, 2),
        "argmax_agreement": round(match, 4),
        "score_max_diff": round(val_diff, 6),
    }


def bench_end_to_end(pipe, image, prompt, keep_ratios, iters=10):
    """token 压缩率 vs TTFT / 精度。这才是项目的主线结论。"""
    out = []
    for kr in keep_ratios:
        pipe.set_token_keep_ratio(kr)          # 需在 run_vl.py 里接上
        lat, ntok = [], None
        for _ in range(iters):
            r = pipe.generate(image, prompt, max_new_tokens=128, stream=False)
            lat.append(r["ttft_ms"])
            ntok = r["n_visual_tokens"]
        lat.sort()
        out.append({
            "keep_ratio": kr,
            "n_visual_tokens": ntok,
            "ttft_p50_ms": round(lat[len(lat)//2], 2),
        })
        print(f"  keep={kr:.2f}  视觉token={ntok:5d}  TTFT={out[-1]['ttft_p50_ms']:8.2f}ms")
    base = out[0]["ttft_p50_ms"]
    for o in out:
        o["ttft_reduction_pct"] = round((base - o["ttft_p50_ms"]) / base * 100, 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tokens", type=int, default=1024)
    ap.add_argument("--dim", type=int, default=2048)
    ap.add_argument("--sweep", action="store_true", help="扫多个 token 数")
    ap.add_argument("--e2e", action="store_true", help="端到端测试（需 engine 就绪）")
    ap.add_argument("--out", default="results/raw/kernel_bench.json")
    args = ap.parse_args()

    results = {"kernel_level": [], "end_to_end": []}

    sizes = [256, 576, 1024, 1600, 2048] if args.sweep else [args.n_tokens]
    print(f"{'N':>7} {'D':>6} {'融合(ms)':>10} {'朴素(ms)':>10} {'加速':>7} "
          f"{'S矩阵':>9} {'argmax一致':>10}")
    print("-" * 68)
    for n in sizes:
        try:
            r = bench_kernel_level(n, args.dim)
            results["kernel_level"].append(r)
            print(f"{r['n_tokens']:>7} {r['dim']:>6} {r['fused_ms']:>10.4f} "
                  f"{r['naive_ms']:>10.4f} {r['speedup']:>6.2f}x "
                  f"{r['S_matrix_MB']:>8.2f}M {r['argmax_agreement']*100:>9.1f}%")
        except ImportError:
            print("未安装 token_merge_cuda，先 CUDA_ARCH=87 python setup.py install")
            return
        except Exception as e:
            print(f"{n:>7}  失败: {e}")

    if args.e2e:
        print("\n端到端（token 压缩率 vs TTFT）：")
        from runtime.run_vl import VLPipeline
        pipe = VLPipeline("Qwen/Qwen2.5-VL-3B-Instruct",
                          "engines/vit_fp16.engine", "engines/llm_int4awq")
        results["end_to_end"] = bench_end_to_end(
            pipe, "assets/demo.jpg", "详细描述这张图片。",
            keep_ratios=[1.0, 0.9, 0.75, 0.6, 0.5])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")
    print("\n提醒：Nsight Compute 再跑一遍，拿到 dram__throughput 和 SM 利用率，")
    print("      面试问'你怎么知道优化生效了'要用这个答。")
    print("  ncu --set full --kernel-name fused_match_kernel python bench_kernel.py")


if __name__ == "__main__":
    main()
