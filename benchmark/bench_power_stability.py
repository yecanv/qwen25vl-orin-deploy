#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功耗采样与长稳测试
==================

车端 JD 原文里点名的指标：「长稳与抖动」「能效」。这是纯语言项目通常没有、
但车载场景一定会问的东西。

两件事：

1. **功耗 / 能效比**
   Orin 上功耗从 `tegrastats` 读。注意几点：
   - Orin 的 VDD_IN 是整板输入功率（含 CPU/DRAM/风扇），VDD_CPU_GPU_CV 才是
     计算单元功耗。报能效比要写清楚用的哪个口径，不然数字没法比。
   - 采样周期最小 ~50ms，比单个 decode step 还长，所以只能算平均功耗，
     不要谎称测到了 per-token 功耗。
   - 能效比 = 总输出 token 数 / 总能耗(J)，单位 tokens/J。

2. **长稳与热降频**
   Orin 被动散热的板子跑 10 分钟就会到热墙，GPU 频率往下掉，延迟随之上涨。
   这是车载环境的真实问题，也是最能体现"跑过真机"的数据。
   本脚本记录整个过程的 P50/P95/P99 延迟随时间的漂移。

用法：
  python benchmark/bench_power_stability.py --duration 1800 --interval 2
"""

import argparse
import json
import re
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


# --------------------------------------------------------------------------- #
# tegrastats 解析
# --------------------------------------------------------------------------- #

# JetPack 6.x tegrastats 输出样例（字段随版本变化，解析要容错）：
# RAM 5432/7620MB ... GR3D_FREQ 99% ... cpu@52.5C soc0@50.2C tj@53.1C
# VDD_IN 8234mW/7891mW VDD_CPU_GPU_CV 4102mW/3980mW VDD_SOC 1876mW/1802mW

RE_RAM   = re.compile(r"RAM (\d+)/(\d+)MB")
RE_SWAP  = re.compile(r"SWAP (\d+)/(\d+)MB")
RE_GR3D  = re.compile(r"GR3D_FREQ (\d+)%")
RE_TJ    = re.compile(r"tj@([\d.]+)C")
RE_VDDIN = re.compile(r"VDD_IN (\d+)mW/(\d+)mW")
RE_VDDGPU= re.compile(r"VDD_CPU_GPU_CV (\d+)mW/(\d+)mW")
RE_VDDSOC= re.compile(r"VDD_SOC (\d+)mW/(\d+)mW")


@dataclass
class PowerSample:
    t: float
    ram_mb: int = 0
    ram_total_mb: int = 0
    swap_mb: int = 0
    gpu_util: int = 0
    tj_c: float = 0.0
    vdd_in_mw: int = 0
    vdd_gpu_mw: int = 0
    vdd_soc_mw: int = 0


def parse_tegrastats_line(line: str, t: float) -> Optional[PowerSample]:
    s = PowerSample(t=t)
    m = RE_RAM.search(line)
    if not m:
        return None
    s.ram_mb, s.ram_total_mb = int(m.group(1)), int(m.group(2))
    if (m := RE_SWAP.search(line)):   s.swap_mb = int(m.group(1))
    if (m := RE_GR3D.search(line)):   s.gpu_util = int(m.group(1))
    if (m := RE_TJ.search(line)):     s.tj_c = float(m.group(1))
    if (m := RE_VDDIN.search(line)):  s.vdd_in_mw = int(m.group(1))
    if (m := RE_VDDGPU.search(line)): s.vdd_gpu_mw = int(m.group(1))
    if (m := RE_VDDSOC.search(line)): s.vdd_soc_mw = int(m.group(1))
    return s


class TegrastatsMonitor:
    """后台线程持续采样，主线程跑推理。"""

    def __init__(self, interval_ms: int = 200):
        self.interval_ms = interval_ms
        self.samples: List[PowerSample] = []
        self._proc = None
        self._thread = None
        self._stop = threading.Event()

    def _run(self):
        self._proc = subprocess.Popen(
            ["tegrastats", "--interval", str(self.interval_ms)],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        t0 = time.time()
        for line in self._proc.stdout:
            if self._stop.is_set():
                break
            s = parse_tegrastats_line(line, time.time() - t0)
            if s:
                self.samples.append(s)

    def __enter__(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        time.sleep(1.0)   # 等第一批采样
        if not self.samples:
            print("[warn] tegrastats 无输出。检查：是否在 Jetson 上运行？"
                  "是否需要 sudo？功耗数据将缺失。")
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._proc:
            self._proc.terminate()
        if self._thread:
            self._thread.join(timeout=3)

    def energy_joules(self, t_start: float, t_end: float,
                      rail: str = "vdd_in") -> float:
        """梯形积分算区间能耗。"""
        key = {"vdd_in": "vdd_in_mw", "gpu": "vdd_gpu_mw",
               "soc": "vdd_soc_mw"}[rail]
        pts = [(s.t, getattr(s, key)) for s in self.samples
               if t_start <= s.t <= t_end]
        if len(pts) < 2:
            return float("nan")
        e = 0.0
        for (t1, p1), (t2, p2) in zip(pts, pts[1:]):
            e += (p1 + p2) / 2 * (t2 - t1) / 1000.0   # mW*s → J
        return e


# --------------------------------------------------------------------------- #
# 长稳测试
# --------------------------------------------------------------------------- #

@dataclass
class StabilityResult:
    latencies_ms: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    n_output_tokens: List[int] = field(default_factory=list)
    errors: int = 0


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def run_stability(pipe, image: str, prompt: str, duration_s: int,
                  monitor: TegrastatsMonitor,
                  window_s: int = 60) -> Dict:
    res = StabilityResult()
    t0 = time.time()
    print(f"[stab] 开始 {duration_s}s 连续压测…")

    while time.time() - t0 < duration_s:
        try:
            ts = time.time() - t0
            r = pipe.generate(image, prompt, max_new_tokens=128, stream=False)
            res.latencies_ms.append(r["total_ms"])
            res.timestamps.append(ts)
            res.n_output_tokens.append(r["n_output_tokens"])
        except Exception as e:
            res.errors += 1
            print(f"[stab] 请求失败 ({res.errors}): {e}")

        el = time.time() - t0
        if len(res.latencies_ms) % 20 == 0:
            recent = res.latencies_ms[-20:]
            print(f"[stab] {el:6.0f}s  n={len(res.latencies_ms):4d}  "
                  f"p50={percentile(recent,0.5):7.1f}ms  "
                  f"tj={monitor.samples[-1].tj_c if monitor.samples else 0:.1f}C")

    # ---- 分窗统计，看漂移 ----
    windows = []
    n_win = max(1, int(duration_s // window_s))
    for i in range(n_win):
        lo, hi = i * window_s, (i + 1) * window_s
        lat = [l for l, t in zip(res.latencies_ms, res.timestamps) if lo <= t < hi]
        if not lat:
            continue
        temps = [s.tj_c for s in monitor.samples if lo <= s.t < hi]
        gpu = [s.gpu_util for s in monitor.samples if lo <= s.t < hi]
        windows.append({
            "window": f"{lo}-{hi}s",
            "n": len(lat),
            "p50_ms": round(percentile(lat, 0.50), 2),
            "p95_ms": round(percentile(lat, 0.95), 2),
            "p99_ms": round(percentile(lat, 0.99), 2),
            "tj_mean_c": round(sum(temps) / len(temps), 1) if temps else None,
            "gpu_util_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
        })

    total_tokens = sum(res.n_output_tokens)
    energy_j = monitor.energy_joules(0, duration_s, "vdd_in")
    energy_gpu_j = monitor.energy_joules(0, duration_s, "gpu")

    # 漂移量：首窗 vs 末窗 p50，直接反映热降频影响
    drift = None
    if len(windows) >= 2:
        drift = round(
            (windows[-1]["p50_ms"] - windows[0]["p50_ms"])
            / windows[0]["p50_ms"] * 100, 2)

    return {
        "duration_s": duration_s,
        "n_requests": len(res.latencies_ms),
        "n_errors": res.errors,
        "overall": {
            "p50_ms": round(percentile(res.latencies_ms, 0.50), 2),
            "p95_ms": round(percentile(res.latencies_ms, 0.95), 2),
            "p99_ms": round(percentile(res.latencies_ms, 0.99), 2),
            "max_ms": round(max(res.latencies_ms), 2) if res.latencies_ms else None,
        },
        "windows": windows,
        "p50_drift_pct": drift,
        "thermal": {
            "tj_max_c": round(max((s.tj_c for s in monitor.samples), default=0), 1),
            "tj_mean_c": round(sum(s.tj_c for s in monitor.samples)
                               / max(len(monitor.samples), 1), 1),
        },
        "memory": {
            "ram_peak_mb": max((s.ram_mb for s in monitor.samples), default=0),
            "ram_total_mb": monitor.samples[0].ram_total_mb if monitor.samples else 0,
            "swap_peak_mb": max((s.swap_mb for s in monitor.samples), default=0),
        },
        "power": {
            "vdd_in_mean_mw": round(sum(s.vdd_in_mw for s in monitor.samples)
                                    / max(len(monitor.samples), 1), 1),
            "vdd_in_peak_mw": max((s.vdd_in_mw for s in monitor.samples), default=0),
            "energy_total_j": round(energy_j, 1),
            "energy_gpu_j": round(energy_gpu_j, 1),
        },
        "efficiency": {
            "total_output_tokens": total_tokens,
            "tokens_per_joule_vddin": round(total_tokens / energy_j, 3)
                                      if energy_j and energy_j == energy_j else None,
            "tokens_per_joule_gpu": round(total_tokens / energy_gpu_j, 3)
                                    if energy_gpu_j and energy_gpu_j == energy_gpu_j else None,
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--vit-engine", default="engines/vit_fp16.engine")
    ap.add_argument("--llm-engine", default="engines/llm_int4awq")
    ap.add_argument("--image", default="assets/demo.jpg")
    ap.add_argument("--prompt", default="描述这张图片。")
    ap.add_argument("--duration", type=int, default=1800, help="秒，建议 ≥1800")
    ap.add_argument("--interval", type=int, default=200, help="tegrastats 采样 ms")
    ap.add_argument("--out", default="results/raw/stability.json")
    args = ap.parse_args()

    from runtime.run_vl import VLPipeline

    pipe = VLPipeline(args.model, args.vit_engine, args.llm_engine)

    # 预热，排除首次 kernel autotuning 的影响
    print("[warm] 预热 5 次…")
    for _ in range(5):
        pipe.generate(args.image, args.prompt, max_new_tokens=32, stream=False)

    with TegrastatsMonitor(args.interval) as mon:
        result = run_stability(pipe, args.image, args.prompt,
                               args.duration, mon)

    result["cold_start_s"] = round(pipe.cold_start_s, 2)
    result["config"] = vars(args)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 62)
    print(f"请求数 {result['n_requests']}  错误 {result['n_errors']}")
    print(f"P50/P95/P99 = {result['overall']['p50_ms']} / "
          f"{result['overall']['p95_ms']} / {result['overall']['p99_ms']} ms")
    print(f"P50 漂移 {result['p50_drift_pct']}%   "
          f"结温峰值 {result['thermal']['tj_max_c']}C")
    print(f"内存峰值 {result['memory']['ram_peak_mb']}/"
          f"{result['memory']['ram_total_mb']} MB")
    print(f"平均整板功耗 {result['power']['vdd_in_mean_mw']/1000:.2f} W")
    print(f"能效比 {result['efficiency']['tokens_per_joule_vddin']} tokens/J (VDD_IN)")
    print(f"\n原始数据 → {out}")


if __name__ == "__main__":
    main()
