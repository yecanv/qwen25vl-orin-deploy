#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上验证：TRT engine vs 桌面 PyTorch 黄金参考（对拍 + 测时延）
用法: python3 vit_orin_verify.py  （在 ~/work/qwen25vl-orin-deploy 下运行）"""
import sys, os, time, json
import numpy as np

sys.path.insert(0, os.path.expanduser("~/work/qwen25vl-orin-deploy"))
from runtime.run_vl import VitEngine   # 复用仓库的 TRT10 封装

G = np.load(os.path.expanduser("~/work/qwen25vl-orin-deploy/results/vit_golden_4096.npz"))
pixels = G["pixel_values"]            # fp16 [4096,1176]
grid   = G["grid_thw"]                # int64 [1,3]
ref    = G["ref_output"]              # fp32 [1024,2048]
print(f"golden: pixels{pixels.shape} grid{grid.tolist()} ref{ref.shape}")

eng = VitEngine(os.path.expanduser("~/work/qwen25vl-orin-deploy/engines/vit_fp16.engine"))

# ---- 正确性：与桌面 PyTorch 黄金输出比 ----
out = eng(pixels, grid)
a = out.astype(np.float64).ravel()
b = ref.astype(np.float64).ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
mad = float(np.abs(a - b).max())
print(f"[对拍] 余弦相似度 = {cos:.6f}   最大绝对误差 = {mad:.5f}")

# ---- 时延：warmup 20 + 100 次计时 ----
for _ in range(20):
    eng(pixels, grid)
lat = []
for _ in range(100):
    t0 = time.perf_counter()
    eng(pixels, grid)
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
stats = {
    "cos_vs_desktop_pytorch": round(cos, 6),
    "max_abs_diff": round(mad, 5),
    "latency_ms": {"mean": round(sum(lat)/len(lat), 2),
                   "p50": round(lat[49], 2), "p95": round(lat[94], 2),
                   "min": round(lat[0], 2), "max": round(lat[-1], 2)},
    "shape": "4096 patch / 1024 visual token (~896x896)",
    "power_mode": "MAXN_SUPER + jetson_clocks",
}
print(f"[时延] mean={stats['latency_ms']['mean']}ms  p50={stats['latency_ms']['p50']}ms  "
      f"p95={stats['latency_ms']['p95']}ms")
outp = os.path.expanduser("~/work/qwen25vl-orin-deploy/results/raw/vit_orin_verify.json")
json.dump(stats, open(outp, "w"), ensure_ascii=False, indent=2)
print(f"→ {outp}")
assert cos > 0.999, "与桌面参考不一致，engine 有问题"
print("验证通过：engine 正确且拿到 Orin 实测时延")
