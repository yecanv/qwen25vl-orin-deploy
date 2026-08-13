#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""板上验证 v3：全静态 engine（单输入/静态输出，无需 OutputAllocator）"""
import os, time, json
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

ROOT = os.path.expanduser("~/work/qwen25vl-orin-deploy")
G = np.load(f"{ROOT}/results/vit_golden_4096.npz")
pixels, ref = G["pixel_values"], G["ref_output"]

logger = trt.Logger(trt.Logger.WARNING)
with open(f"{ROOT}/engines/vit_fp16_static.engine", "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
stream = cuda.Stream()

out_shape = tuple(ctx.get_tensor_shape("vision_embeds"))
print(f"静态输出形状: {out_shape}")
assert all(d > 0 for d in out_shape), "静态 engine 不应有动态维"

d_pix = cuda.mem_alloc(pixels.nbytes)
out = np.empty(out_shape, dtype=np.float16)
d_out = cuda.mem_alloc(out.nbytes)
cuda.memcpy_htod(d_pix, np.ascontiguousarray(pixels))
ctx.set_tensor_address("pixel_values", int(d_pix))
ctx.set_tensor_address("vision_embeds", int(d_out))

def infer():
    ctx.execute_async_v3(stream.handle)
    stream.synchronize()

infer()
cuda.memcpy_dtoh(out, d_out)
a = out.astype(np.float64).ravel()
b = ref.astype(np.float64).ravel()
cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
mad = float(np.abs(a - b).max())
print(f"[对拍] 余弦相似度 = {cos:.6f}   最大绝对误差 = {mad:.5f}")

for _ in range(20):
    infer()
lat = []
for _ in range(100):
    t0 = time.perf_counter()
    infer()
    lat.append((time.perf_counter() - t0) * 1000)
lat.sort()
stats = {
    "cos_vs_desktop_pytorch": round(cos, 6),
    "max_abs_diff": round(mad, 5),
    "latency_ms": {"mean": round(sum(lat)/len(lat), 2), "p50": round(lat[49], 2),
                   "p95": round(lat[94], 2), "min": round(lat[0], 2)},
    "engine": "vit_fp16_static.engine (grid常量化全静态, builderOptLevel=3)",
    "shape": "4096 patch / 1024 visual token (~896x896)",
    "power_mode": "MAXN_SUPER + jetson_clocks",
    "note": "纯 GPU 计算时延(不含H2D/D2H, 输入常驻显存)",
}
print(f"[时延] mean={stats['latency_ms']['mean']}ms p50={stats['latency_ms']['p50']}ms "
      f"p95={stats['latency_ms']['p95']}ms min={stats['latency_ms']['min']}ms")
json.dump(stats, open(f"{ROOT}/results/raw/vit_orin_verify_static.json", "w"),
          ensure_ascii=False, indent=2)
print("✅ 通过" if cos > 0.999 else "❌ 仍不一致")
