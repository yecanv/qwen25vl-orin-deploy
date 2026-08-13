#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多图 M-RoPE 位置编码对拍:runtime/mrope.py vs HF get_rope_index(逐元素)
桌面运行(qwen_trt 环境),不加载权重——get_rope_index 是纯位置数学,只需 config。
双图输入用真实 processor 构造(两张合成图),覆盖:
  文本段 + 图A(896²,网格64x64) + 文本段 + 图B(448²,网格32x32) + 文本段
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image, ImageDraw

MODEL_DIR = r"E:\models\qwen25vl-3b"

def make_imgs():
    a = Image.new("RGB", (896, 896), "skyblue")           # 图A:城市方块场景
    d = ImageDraw.Draw(a)
    for x in (100, 400, 700):
        d.rectangle([x, 300, x + 120, 800], fill="gray")
    b = Image.new("RGB", (448, 448), "white")             # 图B:红圆+绿方块
    d = ImageDraw.Draw(b)
    d.ellipse([124, 124, 324, 324], fill="red")
    d.rectangle([20, 20, 100, 100], fill="green")
    return a, b

def main():
    import torch
    from transformers import AutoProcessor, AutoConfig

    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    img_a, img_b = make_imgs()
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": "先描述第一张。"},
        {"type": "image"}, {"type": "text", "text": "再描述第二张,两张分开说。"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_a, img_b], return_tensors="pt")
    input_ids = inputs["input_ids"]                       # [1, seq]
    grid = inputs["image_grid_thw"]                       # [2, 3]
    print("双图输入构造完成: seq_len =", input_ids.shape[1], " grid_thw =", grid.tolist())

    # ---- HF 参考:不加载权重,空壳模型只为 get_rope_index ----
    from accelerate import init_empty_weights
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(cfg)
    fn = getattr(model, "get_rope_index", None) or getattr(model.model, "get_rope_index")
    ref_pos, ref_delta = fn(input_ids, grid, None, attention_mask=torch.ones_like(input_ids))
    ref_pos = ref_pos[:, 0, :].numpy()                    # [3, seq]
    ref_delta = int(ref_delta.flatten()[0])

    # ---- 我们的实现 ----
    from runtime.mrope import compute_mrope_position_ids
    my_pos, my_delta = compute_mrope_position_ids(
        input_ids[0].numpy(), grid.numpy())

    same = np.array_equal(my_pos, ref_pos)
    print(f"position_ids 逐元素一致: {same}   形状 {my_pos.shape} vs {ref_pos.shape}")
    print(f"mrope_delta: 我们 {my_delta}  HF {ref_delta}  一致: {my_delta == ref_delta}")
    if not same:
        diff = np.argwhere(my_pos != ref_pos)
        print("首个不一致位置:", diff[:5].tolist())
        for r, c in diff[:5]:
            print(f"  [{r},{c}] 我们={my_pos[r,c]} HF={ref_pos[r,c]}")
        sys.exit(1)
    # 展示隔离结构:两段图像的 t 分量取值
    ids = input_ids[0].numpy()
    from runtime.mrope import IMAGE_PAD_ID
    pad_pos = np.where(ids == IMAGE_PAD_ID)[0]
    n_a = int(grid[0, 1] * grid[0, 2] // 4)
    t_a = set(my_pos[0, pad_pos[:n_a]].tolist())
    t_b = set(my_pos[0, pad_pos[n_a:]].tolist())
    print(f"图A t分量取值 {t_a}  图B t分量取值 {t_b}  段号隔离: {t_a != t_b}")
    print("MULTI_IMG_MROPE_CHECK_PASS")

if __name__ == "__main__":
    main()
