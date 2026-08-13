#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多图管线胶水单测(桌面,无引擎无 LLM):
真 processor 构造双图输入,伪 ViT/伪压缩注入,验证 generate() 前半段的
全部形状与下标契约——逐图切片、按图分段压缩、shrink/pos 裁剪对齐、
fake id 连续性、断言链不触发。伪 ViT 的输出行值编码"图号+图内下标",
最后反查 prompt table 每行确实来自正确的图与位置(压缩不跨图的铁证)。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from runtime.mrope import (compute_mrope_position_ids, build_prompt_table,
                           IMAGE_PAD_ID)
from runtime.mrope_multiimg_check import make_imgs, MODEL_DIR


def fake_vit(pix, grid):
    """伪 ViT:输出 [n_tok, 4],每行 = [图号, 图内 token 下标, n_patch, 0]
    图号用 n_patch 反推不了,由调用方经闭包计数。"""
    n_patch = pix.shape[0]
    n_tok = n_patch // 4
    i = fake_vit.call_count
    fake_vit.call_count += 1
    out = np.zeros((n_tok, 4), dtype=np.float16)
    out[:, 0] = i                       # 图号
    out[:, 1] = np.arange(n_tok)        # 图内下标
    out[:, 2] = n_patch
    return out
fake_vit.call_count = 0


def fake_compress(vis):
    """伪压缩:确定性丢掉本段第 1、3 个 token(模拟散布存活)"""
    n = vis.shape[0]
    keep = np.ones(n, dtype=bool)
    if n > 4:
        keep[[1, 3]] = False
    idx = np.nonzero(keep)[0]
    return vis[idx], idx


def main():
    from transformers import AutoProcessor
    proc = AutoProcessor.from_pretrained(MODEL_DIR)
    img_a, img_b = make_imgs()
    msgs = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"},
        {"type": "text", "text": "分别描述两张图。"}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img_a, img_b], return_tensors="pt")

    input_ids = inputs["input_ids"][0].numpy()
    input_ids_raw = input_ids.copy()
    grid = inputs["image_grid_thw"].numpy()
    pix = inputs["pixel_values"].numpy().astype(np.float16)

    # ---- 复刻 generate() 阶段 1:逐图前向 ----
    n_patches = [int(t * h * w) for t, h, w in grid.tolist()]
    vis_parts, off = [], 0
    for i, n_p in enumerate(n_patches):
        vis_parts.append(fake_vit(pix[off:off + n_p], grid[i:i + 1]))
        off += n_p
    vis = np.concatenate(vis_parts, axis=0)
    assert off == pix.shape[0], "patch 切分未覆盖全部行"

    # ---- 复刻阶段 1.5:按图分段压缩 ----
    n_vis_raw = int(vis.shape[0])
    seg_sizes = [n // 4 for n in n_patches]
    parts, idx_parts, base = [], [], 0
    for seg in seg_sizes:
        seg_vis, seg_idx = fake_compress(vis[base:base + seg])
        parts.append(seg_vis)
        idx_parts.append(seg_idx + base)
        base += seg
    vis_c = np.concatenate(parts, axis=0)
    kept_vis_idx = np.concatenate(idx_parts)

    # 契约①:压缩不跨图——每段存活行的"图号"列必须等于本段图号
    base = 0
    for i, seg in enumerate(seg_sizes):
        seg_rows = vis_c[(kept_vis_idx >= base) & (kept_vis_idx < base + seg)]
        assert (seg_rows[:, 0] == i).all(), f"图 {i} 的段里混入了其他图的 token!"
        base += seg
    print(f"契约① 压缩不跨图: PASS  (每图丢 2 个,存活 {vis_c.shape[0]}/{n_vis_raw})")

    # ---- 复刻 shrink + 桥接 ----
    pos = np.where(input_ids == IMAGE_PAD_ID)[0]
    assert len(pos) == n_vis_raw, "占位符数 != 视觉 token 数"
    keep = np.ones(len(input_ids), dtype=bool)
    dropped = np.setdiff1d(np.arange(n_vis_raw), kept_vis_idx)
    keep[pos[dropped]] = False
    ids_shrunk = input_ids[keep]

    pos_ids_full, _ = compute_mrope_position_ids(input_ids_raw, grid)
    pos_ids = pos_ids_full[:, keep]
    delta = int(pos_ids.max()) + 1 - pos_ids.shape[1]

    fake_ids, ptable = build_prompt_table(ids_shrunk, vis_c, 151936)

    # 契约②:形状对齐链
    assert pos_ids.shape[1] == len(fake_ids) == len(ids_shrunk)
    assert ptable.shape[0] == vis_c.shape[0]
    print(f"契约② 形状对齐: PASS  (序列 {len(fake_ids)},表 {ptable.shape[0]} 行,delta={delta})")

    # 契约③:prompt table 每行反查——第 k 个存活视觉 token 的表行内容
    # 应等于"它原图号 + 原图内下标"(伪 ViT 编码),证明拼接顺序全程未错位
    seg_base = 0
    ok = True
    for i, seg in enumerate(seg_sizes):
        m = (kept_vis_idx >= seg_base) & (kept_vis_idx < seg_base + seg)
        rows = ptable[m]
        in_img = kept_vis_idx[m] - seg_base
        ok &= (rows[:, 0] == i).all() and (rows[:, 1] == in_img).all()
        seg_base += seg
    assert ok, "prompt table 行内容与图号/图内下标不符——拼接顺序错位!"
    print("契约③ prompt table 逐行溯源: PASS")
    print("MULTIIMG_GLUE_TEST_PASS")


if __name__ == "__main__":
    main()
