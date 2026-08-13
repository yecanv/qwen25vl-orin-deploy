#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
并发吞吐 Benchmark
==================

VLM 的并发瓶颈和纯语言模型不一样，这是本脚本要测出来的核心结论。

纯语言模型：并发上限由 KV Cache 决定
    max_concurrent ≈ kv_cache_bytes / (2 · layers · kv_heads · head_dim · seq_len · dtype)

VLM：还要额外受 `max_multimodal_len` 约束
    每个请求的视觉 token 都要占 prompt table 的空间，
    build engine 时定死的 max_multimodal_len 会先于 KV Cache 成为瓶颈。

所以本脚本同时记录：
  - 吞吐随并发的变化曲线（找拐点）
  - 每个并发档位的显存占用
  - 失败请求数（OOM 或超出 multimodal_len 会直接报错）

在 Orin 这种统一内存的平台上，还有一层：显存和系统内存共用，
并发上去后系统本身可能先被挤爆。
"""

import argparse
import json
import statistics
import threading
import time
from pathlib import Path
from queue import Queue
from typing import List, Dict


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def read_mem_mb() -> int:
    """Orin 统一内存，直接读 /proc/meminfo 比 torch.cuda 准。"""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.strip().split()[0])
        return (info["MemTotal"] - info["MemAvailable"]) // 1024
    except Exception:
        return 0


def run_concurrency(pipe, image: str, prompt: str, n_concurrent: int,
                    n_requests: int, max_new_tokens: int = 128) -> Dict:
    """
    用线程池打并发。注意 TRT-LLM 的 ModelRunner 本身不是线程安全的，
    这里的并发是**请求级**的排队，真正的 batching 由 engine 的
    inflight batching 完成——这正是要测的东西。
    """
    q: Queue = Queue()
    for _ in range(n_requests):
        q.put(1)

    latencies: List[float] = []
    ttfts: List[float] = []
    out_tokens: List[int] = []
    errors = [0]
    lock = threading.Lock()
    mem_samples: List[int] = []
    stop_mem = threading.Event()

    def mem_monitor():
        while not stop_mem.is_set():
            mem_samples.append(read_mem_mb())
            time.sleep(0.2)

    def worker():
        while True:
            try:
                q.get_nowait()
            except Exception:
                return
            try:
                t0 = time.perf_counter()
                r = pipe.generate(image, prompt,
                                  max_new_tokens=max_new_tokens, stream=False)
                dt = (time.perf_counter() - t0) * 1000
                with lock:
                    latencies.append(dt)
                    ttfts.append(r["ttft_ms"])
                    out_tokens.append(r["n_output_tokens"])
            except Exception as e:
                with lock:
                    errors[0] += 1
                    if errors[0] <= 2:
                        print(f"      请求失败: {str(e)[:120]}")
            finally:
                q.task_done()

    mt = threading.Thread(target=mem_monitor, daemon=True)
    mt.start()

    t_start = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(n_concurrent)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    stop_mem.set()
    mt.join(timeout=1)

    total_out = sum(out_tokens)
    return {
        "concurrency": n_concurrent,
        "n_requests": n_requests,
        "n_success": len(latencies),
        "n_errors": errors[0],
        "wall_s": round(wall, 2),
        "throughput_tok_s": round(total_out / wall, 2) if wall > 0 else 0,
        "throughput_req_s": round(len(latencies) / wall, 3) if wall > 0 else 0,
        "ttft_p50_ms": round(percentile(ttfts, 0.50), 2),
        "ttft_p95_ms": round(percentile(ttfts, 0.95), 2),
        "ttft_p99_ms": round(percentile(ttfts, 0.99), 2),
        "latency_p50_ms": round(percentile(latencies, 0.50), 2),
        "latency_p99_ms": round(percentile(latencies, 0.99), 2),
        "mem_peak_mb": max(mem_samples) if mem_samples else 0,
        "mem_delta_mb": (max(mem_samples) - min(mem_samples)) if mem_samples else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vit-engine", default="engines/vit_fp16.engine")
    ap.add_argument("--llm-engine", default="engines/llm_int4awq")
    ap.add_argument("--image", default="assets/demo.jpg")
    ap.add_argument("--prompt", default="描述这张图片。")
    ap.add_argument("--levels", default="1,2,4,8",
                    help="并发档位。8GB 板子建议到 4 即可")
    ap.add_argument("--requests-per-level", type=int, default=24)
    ap.add_argument("--tag", default="int4awq")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    from runtime.run_vl import VLPipeline

    pipe = VLPipeline(args.model, args.vit_engine, args.llm_engine)

    print("[warm] 预热…")
    for _ in range(3):
        pipe.generate(args.image, args.prompt, max_new_tokens=32, stream=False)

    base_mem = read_mem_mb()
    print(f"[warm] 基线内存占用 {base_mem} MB\n")

    levels = [int(x) for x in args.levels.split(",")]
    results = []
    print(f"{'并发':>5} {'成功/总':>9} {'吞吐(tok/s)':>13} {'TTFT p50':>10} "
          f"{'TTFT p99':>10} {'内存峰值':>10} {'错误':>5}")
    print("-" * 70)

    for c in levels:
        r = run_concurrency(pipe, args.image, args.prompt, c,
                            args.requests_per_level)
        r["mem_over_baseline_mb"] = r["mem_peak_mb"] - base_mem
        results.append(r)
        print(f"{c:>5} {r['n_success']:>4}/{r['n_requests']:<4} "
              f"{r['throughput_tok_s']:>13.2f} {r['ttft_p50_ms']:>10.2f} "
              f"{r['ttft_p99_ms']:>10.2f} {r['mem_peak_mb']:>9d}M {r['n_errors']:>5}")
        if r["n_errors"] > r["n_requests"] * 0.2:
            print(f"      并发 {c} 失败率过高，停止上探")
            print(f"      常见原因：max_multimodal_len 或 max_batch_size 不够，"
                  f"见 convert/build_llm_engine.sh")
            break
        time.sleep(20)   # 降温，避免上一档余热影响下一档

    # 找吞吐拐点
    if len(results) >= 2:
        best = max(results, key=lambda r: r["throughput_tok_s"])
        print(f"\n吞吐峰值出现在并发 {best['concurrency']}："
              f"{best['throughput_tok_s']} tok/s")
        scaling = [round(r["throughput_tok_s"] / results[0]["throughput_tok_s"], 2)
                   for r in results]
        print(f"相对单路的加速比：{scaling}")
        print("（理想线性是 [1,2,4,8]，实际打折的部分就是 batching 的开销和瓶颈）")

    out = Path(args.out or f"results/raw/throughput_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "tag": args.tag,
        "baseline_mem_mb": base_mem,
        "results": results,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
