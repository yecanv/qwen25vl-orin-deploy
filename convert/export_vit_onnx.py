#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉编码器 ONNX 导出（Qwen2-VL / Qwen2.5-VL）
============================================

Qwen2-VL 系列的 shape 契约（必须严格满足，错一个就报错或静默出错）
-----------------------------------------------------------------

设 grid_thw = (t, h, w)，其中 h、w 是 **merger 之前**的 patch 网格尺寸：

    pixel_values.shape == (t * h * w, 1176)

    1176 = 3(RGB) × 2(temporal_patch_size) × 14 × 14(patch)

    merger 做 2×2 空间合并，所以：
        h % 2 == 0 且 w % 2 == 0        ← 硬约束
        n_visual_tokens = t * (h//2) * (w//2)

    对应原图尺寸：H = h*14, W = w*14，且必须是 28 (=14×2) 的整数倍

上一版的 bug（本次修复）
-----------------------
上一版用 `h = int(n**0.5)//2*2; w = n//h` 去凑 dummy，
结果 t*h*w != pixel_values 的行数。ViT 内部按 grid_thw 重算 cu_seqlens，
和实际行数对不上 —— 报 index out of range，或者更糟：静默错位。

本版改为 **先定视觉 token 数，反推自洽的 (t,h,w) 和 n_patches**，
构造完立刻 assert 校验，并把契约写进 json 供 build 脚本复用。
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn


PATCH = 14
TEMPORAL_PATCH = 2
MERGE = 2
PATCH_DIM = 3 * TEMPORAL_PATCH * PATCH * PATCH   # 1176


# --------------------------------------------------------------------------- #
# shape 推导（本次修复的核心）
# --------------------------------------------------------------------------- #

@dataclass
class GridSpec:
    t: int
    h: int          # merger 前的 patch 行数
    w: int          # merger 前的 patch 列数

    @property
    def n_patches(self) -> int:
        return self.t * self.h * self.w

    @property
    def n_visual_tokens(self) -> int:
        return self.t * (self.h // MERGE) * (self.w // MERGE)

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.h * PATCH, self.w * PATCH

    def validate(self):
        assert self.h % MERGE == 0, f"h={self.h} 必须能被 {MERGE} 整除"
        assert self.w % MERGE == 0, f"w={self.w} 必须能被 {MERGE} 整除"
        assert self.t >= 1 and self.n_patches > 0


def grid_for_visual_tokens(n_visual: int, t: int = 1,
                           aspect: float = 1.0) -> GridSpec:
    """
    给定期望视觉 token 数，反推自洽的 (t, h, w)。

        n_visual = t * (h/2) * (w/2)
        令 a = h/2, b = w/2 → a*b = n_visual/t, h = 2a, w = 2b

    aspect = a/b，用于构造非方形输入（DocVQA 那种宽扁图）。
    """
    if n_visual % t:
        raise ValueError(f"n_visual({n_visual}) 需能被 t({t}) 整除")
    area = n_visual // t

    best = None
    for a in range(1, area + 1):
        if area % a:
            continue
        b = area // a
        err = abs((a / b) - aspect)
        if best is None or err < best[0]:
            best = (err, a, b)
    _, a, b = best

    g = GridSpec(t=t, h=a * MERGE, w=b * MERGE)
    g.validate()
    if g.n_visual_tokens != n_visual:
        raise RuntimeError(f"反推失败: got {g.n_visual_tokens}, want {n_visual}")
    return g


def make_dummy_input(g: GridSpec, device: str = "cuda", dtype=torch.float16):
    """构造严格自洽的 dummy 输入，并双重校验契约。"""
    g.validate()
    pixel_values = torch.randn(g.n_patches, PATCH_DIM, dtype=dtype, device=device)
    grid_thw = torch.tensor([[g.t, g.h, g.w]], dtype=torch.int64, device=device)

    expect = int(grid_thw.prod(dim=-1).sum())
    if pixel_values.shape[0] != expect:
        raise RuntimeError(
            f"shape 契约违反: pixel_values 有 {pixel_values.shape[0]} 行, "
            f"但 grid_thw 乘积为 {expect}")
    return pixel_values, grid_thw


# --------------------------------------------------------------------------- #
# 导出
# --------------------------------------------------------------------------- #

class VisionTowerWrapper(nn.Module):
    """把 grid_thw 变成显式输入，避免 cu_seqlens 被常量折叠。"""

    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, pixel_values: torch.Tensor,
                grid_thw: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=grid_thw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="onnx/vit_fp16.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--opt-visual-tokens", type=int, default=1024,
                    help="主力分辨率对应的视觉 token 数；1024 ≈ 896x896。"
                         "必须与 build_vit_engine.sh 的 OPT 档一致")
    ap.add_argument("--aspect", type=float, default=1.0)
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq

    g = grid_for_visual_tokens(args.opt_visual_tokens, aspect=args.aspect)
    H, W = g.image_size
    print("[vit] dummy 输入规格（已校验自洽）:")
    print(f"      grid_thw     = ({g.t}, {g.h}, {g.w})")
    print(f"      n_patches    = {g.n_patches}  (= t*h*w)")
    print(f"      视觉 token   = {g.n_visual_tokens}")
    print(f"      等效原图     = {H}x{W}")
    print(f"      pixel_values = [{g.n_patches}, {PATCH_DIM}]")

    print(f"\n[vit] 加载 {args.model}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map="cuda", trust_remote_code=True,
        # 导出用 eager 注意力：新版 transformers 的 SDPA 路径带 enable_gqa
        # 参数，TorchScript 导出器不支持（AssertionError）；且朴素
        # matmul+softmax 图对 TensorRT 更友好——TRT 构建时会自行做注意力
        # 层融合，不需要 ONNX 里出现 SDPA 黑盒算子。
        attn_implementation="eager",
    )
    visual = getattr(model, "visual", None) or getattr(model, "vision_tower", None)
    if visual is None:
        raise RuntimeError(
            "找不到视觉塔。先打印模块名确认属性:\n"
            "  for n, _ in model.named_children(): print(n)")

    # 从 config 读实际 merge_size，别硬编
    try:
        cfg_merge = model.config.vision_config.spatial_merge_size
        if cfg_merge != MERGE:
            print(f"[vit] !! config 的 spatial_merge_size={cfg_merge}，"
                  f"本脚本假设 {MERGE}。请改 MERGE 常量后重跑。")
            return
    except AttributeError:
        print("[vit] 未能从 config 读到 spatial_merge_size，沿用默认 2")

    wrapper = VisionTowerWrapper(visual).eval()
    dummy_pixels, dummy_grid = make_dummy_input(g)

    print("[vit] PyTorch 前向自检…")
    with torch.no_grad():
        ref = wrapper(dummy_pixels, dummy_grid)
    print(f"      输出 shape = {tuple(ref.shape)}")
    if ref.shape[0] != g.n_visual_tokens:
        raise RuntimeError(
            f"输出 token 数 {ref.shape[0]} != 期望 {g.n_visual_tokens}，"
            f"检查 spatial_merge_size")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[vit] 导出 ONNX → {out}")
    # dynamo=False：torch 2.9+ 默认走 dynamo 导出，但 Qwen2.5-VL 的
    # get_window_index 里 cu_window_seqlens 走 .tolist() 数据依赖控制流，
    # dynamo 符号化追踪直接报 GuardOnDataDependentSymNode，只能退回
    # TorchScript trace 路径。
    # ⚠️ 代价：trace 会把窗口索引按本次 dummy 的 grid 烘焙成常量——
    # 同 shape 数值正确，换分辨率会静默出错。所以下面除了 OPT shape
    # 校验，还加了一个"第二形状"校验来显式暴露这一点；
    # 结论以两次校验的实际输出为准。
    torch.onnx.export(
        wrapper,
        (dummy_pixels, dummy_grid),
        str(out),
        input_names=["pixel_values", "grid_thw"],
        output_names=["vision_embeds"],
        dynamic_axes={
            "pixel_values":  {0: "n_patches"},
            "grid_thw":      {0: "n_images"},
            "vision_embeds": {0: "n_visual_tokens"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    # 契约落盘，build 脚本直接读，避免两边手抄对不上
    meta = out.with_suffix(".shapes.json")
    meta.write_text(json.dumps({
        "patch_dim": PATCH_DIM,
        "merge_size": MERGE,
        "hidden": int(ref.shape[-1]),
        "opt": {"t": g.t, "h": g.h, "w": g.w,
                "n_patches": g.n_patches,
                "n_visual_tokens": g.n_visual_tokens},
        "hint": "build_vit_engine.sh 用 opt.n_patches 作 optShapes",
    }, indent=2), encoding="utf-8")
    print(f"[vit] shape 契约 → {meta}")

    if not args.skip_verify:
        print("\n[vit] 校验 1/2：OPT shape（与导出 dummy 同形状）")
        _verify(str(out), dummy_pixels, dummy_grid, ref)

        # 第二形状校验：探测 trace 常量烘焙的实际影响。
        # 若此项 shape 报错或相似度低 → 该 ONNX 是"固定 OPT 分辨率"的，
        # 预处理必须 resize 到 OPT 档；要支持多档分辨率需按档各导一个。
        print("\n[vit] 校验 2/2：第二形状（256 token / 448x448，探测常量烘焙）")
        g2 = grid_for_visual_tokens(256, aspect=args.aspect)
        dummy2_pixels, dummy2_grid = make_dummy_input(g2)
        with torch.no_grad():
            ref2 = wrapper(dummy2_pixels, dummy2_grid)
        try:
            _verify(str(out), dummy2_pixels, dummy2_grid, ref2)
        except Exception as e:
            print(f"[vit] !! 第二形状直接报错：{type(e).__name__}: {e}")
            print("[vit] 结论：ONNX 已被 trace 固定为 OPT 分辨率（预期内），"
                  "部署时预处理统一 resize 到 OPT 档")

    print(f"\n下一步：OPT_PATCHES={g.n_patches} bash convert/build_vit_engine.sh")


def _verify(onnx_path, pixels, grid, torch_ref):
    import numpy as np
    try:
        import onnxruntime as ort
    except ImportError:
        print("[vit] 未装 onnxruntime，跳过校验（强烈建议装）")
        return

    print("[vit] ONNX vs PyTorch 一致性校验…")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # 图是 FP16 的（trace 时模型和 dummy 都是 fp16），输入 dtype 必须保持
    # float16——转 float32 会被 onnxruntime 直接拒收（INVALID_ARGUMENT）
    onnx_out = sess.run(None, {
        "pixel_values": pixels.cpu().numpy(),
        "grid_thw": grid.cpu().numpy(),
    })[0]

    a = onnx_out.astype(np.float64).ravel()
    b = torch_ref.float().cpu().numpy().astype(np.float64).ravel()
    if a.shape != b.shape:
        print(f"[vit] !! shape 不一致 onnx={a.shape} torch={b.shape}")
        return
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    print(f"      余弦相似度 = {cos:.6f}   最大绝对误差 = {np.abs(a-b).max():.5f}")
    if cos < 0.999:
        print("[vit] !! 相似度偏低。排查方向：")
        print("       1. cu_seqlens 被常量折叠 → 确认 wrapper 吃到了 grid_thw")
        print("       2. 2D-RoPE 里有 python 分支 → 改成 torch.where")
        print("       3. opset 太低 → 试 18 / 19")
    else:
        print("[vit] OK")


if __name__ == "__main__":
    main()
