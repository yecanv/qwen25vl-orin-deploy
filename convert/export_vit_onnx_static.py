#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""全静态 ViT 导出：grid_thw 烘焙为常量，单输入/静态输出。

为什么有这个变体（板上实测得出的决策）
--------------------------------------------------
动态版（export_vit_onnx.py）把 grid_thw 做成显式输入，本意是防止
cu_seqlens 被常量折叠。但桌面双形状校验已证明：TorchScript trace 把
窗口索引按 dummy 烘焙成了常量，ONNX 实际是固定分辨率的——grid_thw
输入只剩下驱动若干 Range/Reshape 数据依赖子图的作用。

这些子图在 ONNX Runtime 上结果正确（桌面 cos=0.999961），但
TensorRT 10.3 编译后输出确定性错误（Orin 实测 cos=0.1247；trtexec
冒烟测试亦在同类子图报 RESHAPE_ZERO_IS_PLACEHOLDER 内部错误）。

既然分辨率本就固定，把 grid_thw 一并常量化导出全静态图：
1) 消除 TRT 的数据依赖路径 → 正确性；
2) 输出形状静态化 → 不再需要 IOutputAllocator；
3) 给 TRT 完全静态的优化空间 → 性能。
多档分辨率的正确姿势 = 每档各导一个静态 ONNX/engine。
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from convert.export_vit_onnx import grid_for_visual_tokens, make_dummy_input, PATCH_DIM


class StaticVisionWrapper(nn.Module):
    """grid_thw 以 buffer 形式常量化，导出图只有 pixel_values 一个输入。"""

    def __init__(self, visual, grid_thw: torch.Tensor):
        super().__init__()
        self.visual = visual
        self.register_buffer("grid_thw", grid_thw)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.visual(pixel_values, grid_thw=self.grid_thw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--out", default="onnx/vit_fp16_static.onnx")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--opt-visual-tokens", type=int, default=1024)
    ap.add_argument("--golden", default="", help="可选：黄金npz，导出后用ORT对拍")
    args = ap.parse_args()

    from transformers import AutoModelForVision2Seq
    g = grid_for_visual_tokens(args.opt_visual_tokens)
    print(f"[static] 固定档: grid=({g.t},{g.h},{g.w}) patches={g.n_patches} "
          f"tokens={g.n_visual_tokens}")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        trust_remote_code=True, attn_implementation="eager")
    visual = getattr(model, "visual", None) or getattr(model, "vision_tower", None)

    dummy_pixels, dummy_grid = make_dummy_input(g)
    wrapper = StaticVisionWrapper(visual, dummy_grid).eval()

    with torch.no_grad():
        ref = wrapper(dummy_pixels)
    print(f"[static] 前向自检 输出 {tuple(ref.shape)}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    print(f"[static] 导出 → {out}")
    torch.onnx.export(
        wrapper, (dummy_pixels,), str(out),
        input_names=["pixel_values"], output_names=["vision_embeds"],
        opset_version=args.opset, do_constant_folding=True, dynamo=False)

    meta = out.with_suffix(".shapes.json")
    meta.write_text(json.dumps({
        "static": True, "patch_dim": PATCH_DIM,
        "grid_thw": [g.t, g.h, g.w],
        "input": {"pixel_values": [g.n_patches, PATCH_DIM]},
        "output": {"vision_embeds": [g.n_visual_tokens, int(ref.shape[-1])]},
    }, indent=2), encoding="utf-8")
    print(f"[static] 契约 → {meta}")

    if args.golden:
        import onnxruntime as ort
        G = np.load(args.golden)
        sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
        o = sess.run(None, {"pixel_values": G["pixel_values"]})[0]
        a = o.astype(np.float64).ravel()
        b = G["ref_output"].astype(np.float64).ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"[static] ORT vs 黄金参考 余弦 = {cos:.6f}")
        assert cos > 0.999, "静态导出与参考不一致"
        print("[static] ORT 校验通过")


if __name__ == "__main__":
    main()
