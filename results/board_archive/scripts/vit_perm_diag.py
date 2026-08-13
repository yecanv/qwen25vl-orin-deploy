#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断：错误输出是否为黄金参考的行置换（窗口 reorder 被编译错的特征）"""
import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # noqa

ROOT = os.path.expanduser("~/work/qwen25vl-orin-deploy")
G = np.load(f"{ROOT}/results/vit_golden_4096.npz")
pixels, ref = G["pixel_values"], G["ref_output"].astype(np.float32)

logger = trt.Logger(trt.Logger.WARNING)
with open(f"{ROOT}/engines/vit_fp16_static.engine", "rb") as f, trt.Runtime(logger) as rt:
    engine = rt.deserialize_cuda_engine(f.read())
ctx = engine.create_execution_context()
stream = cuda.Stream()

class OutAlloc(trt.IOutputAllocator):
    def __init__(self):
        super().__init__()
        self.mem, self.size, self.shape = None, 0, None
    def reallocate_output(self, name, memory, size, alignment):
        if size > self.size:
            self.mem = cuda.mem_alloc(size); self.size = size
        return int(self.mem)
    def reallocate_output_async(self, name, memory, size, alignment, s):
        return self.reallocate_output(name, memory, size, alignment)
    def notify_shape(self, name, shape):
        self.shape = tuple(shape)

alloc = OutAlloc()
ctx.set_output_allocator("vision_embeds", alloc)
ctx.set_input_shape("pixel_values", pixels.shape)
d_pix = cuda.mem_alloc(pixels.nbytes)
cuda.memcpy_htod(d_pix, np.ascontiguousarray(pixels))
ctx.set_tensor_address("pixel_values", int(d_pix))
ctx.execute_async_v3(stream.handle)
stream.synchronize()
out = np.empty(alloc.shape, dtype=np.float16)
cuda.memcpy_dtoh(out, alloc.mem)
out = out.astype(np.float32)
np.save(f"{ROOT}/results/raw/vit_orin_wrong_output.npy", out)

# 逐行归一化后算余弦矩阵，找每行最近邻
def norm_rows(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)

S = norm_rows(out) @ norm_rows(ref).T          # [1024,1024]
nn_idx = S.argmax(axis=1)
nn_cos = S.max(axis=1)
uniq = len(set(nn_idx.tolist()))
print(f"逐行最近邻: 平均cos={nn_cos.mean():.4f}  中位={np.median(nn_cos):.4f}  "
      f"cos>0.99的行={int((nn_cos>0.99).sum())}/1024")
print(f"最近邻是否构成双射: 唯一目标数={uniq}/1024")
print(f"恒等映射的行数(没被移动)={int((nn_idx==np.arange(1024)).sum())}")
print(f"前16行的映射: {nn_idx[:16].tolist()}")

if (nn_cos > 0.99).sum() > 900 and uniq > 900:
    perm_cos = float((norm_rows(out).ravel() @ norm_rows(ref[nn_idx]).ravel()) / 1024)
    print(f"按置换重排后整体行cos均值≈{nn_cos.mean():.4f}")
    print("✅ 判定：输出是参考的行置换 → 窗口 reorder/逆reorder 被 TRT 编译错")
else:
    print("❌ 不是干净的行置换 → 数值内容本身坏了（指向精度/其他编译错误）")
