#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
M-RoPE 位置编码与 prompt table 构建
===================================

这个文件是整条链路里最容易出错、也最值得在面试里讲的部分。

背景
----
标准 LLM 用 1D RoPE：token 在序列里的位置就是一个标量 p，
旋转角度 θ_i = p / base^(2i/d)。

Qwen2-VL / Qwen2.5-VL 用 **M-RoPE（Multimodal RoPE）**，把位置拆成三个分量：

    position_id = (t, h, w)     # temporal, height, width

- **文本 token**：三个分量取相同值，退化为普通 1D RoPE
      text token 第 k 个 → (k, k, k)
- **图像 token**：t 固定（单图为 0），h/w 按 patch 在二维网格里的坐标展开
      grid 为 H×W 时，第 (i,j) 个 patch → (t0, i, j)
- **视频**：t 随帧递增

头维度按 mrope_section 划分（Qwen2-VL 默认 [16, 24, 24]，对应 t/h/w 各占的
rotary 维度数），三段各自用对应分量算旋转。

为什么这是最容易出的 bug
------------------------
1. **算错不报错**。position_ids 形状对得上就能跑，只是输出乱码或者
   "看图说瞎话"——图里明明是红灯，模型说是绿灯。很容易误判成模型能力问题
   或者量化掉点，实际是位置编码错了。
2. 图像 token 之后的文本 token，其起始位置**不是简单累加序列长度**，
   而是要接在 max(h, w) 之后。这条最常写错。
3. 多图输入时，每张图的 t 要递增，且后续文本要接在所有图之后。

调试方法（写在这里，别到时候抓瞎）
--------------------------------
- 先用 HF transformers 原版跑一遍，把 `model.get_rope_index()` 的输出 dump 出来
- 再用本文件的实现算一遍，逐元素 diff
- 完全一致才往下走。不一致就单独构造一张 2x2 patch 的小图手推
"""

from typing import List, Tuple, Optional
import numpy as np


# Qwen2-VL / 2.5-VL 默认配置
VISION_START_ID = 151652   # <|vision_start|>
VISION_END_ID   = 151653   # <|vision_end|>
IMAGE_PAD_ID    = 151655   # <|image_pad|>
VIDEO_PAD_ID    = 151656   # <|video_pad|>
MROPE_SECTION   = [16, 24, 24]
SPATIAL_MERGE   = 2        # patch merger 2x2


def compute_mrope_position_ids(
    input_ids: np.ndarray,
    image_grid_thw: Optional[np.ndarray] = None,
    video_grid_thw: Optional[np.ndarray] = None,
    spatial_merge_size: int = SPATIAL_MERGE,
) -> Tuple[np.ndarray, int]:
    """
    参数
    ----
    input_ids       : [seq_len]，已包含 image_pad 占位 token
    image_grid_thw  : [n_images, 3]，每张图 merger **之前** 的 (t, h, w)

    返回
    ----
    position_ids : [3, seq_len]，三行分别是 t / h / w
    mrope_delta  : int，用于 decode 阶段续算位置（decode 时新 token 的位置
                   = 当前长度 + mrope_delta，这个 delta 必须保存下来，
                   否则生成到第二个 token 就错位）

    实现对齐 transformers 的 Qwen2VLForConditionalGeneration.get_rope_index()
    """
    seq_len = len(input_ids)
    pos = np.zeros((3, seq_len), dtype=np.int64)

    img_idx = 0
    vid_idx = 0
    cur = 0          # 当前已消耗到的位置基准
    i = 0

    while i < seq_len:
        tok = input_ids[i]

        if tok == IMAGE_PAD_ID and image_grid_thw is not None:
            t, h, w = image_grid_thw[img_idx]
            h_m = h // spatial_merge_size
            w_m = w // spatial_merge_size
            n_tok = int(t * h_m * w_m)

            # t 分量：单图恒为 cur；多帧视频则随帧递增
            t_idx = np.repeat(np.arange(t), h_m * w_m)
            # h 分量：行坐标，每行重复 w_m 次
            h_idx = np.tile(np.repeat(np.arange(h_m), w_m), t)
            # w 分量：列坐标
            w_idx = np.tile(np.arange(w_m), t * h_m)

            pos[0, i:i + n_tok] = t_idx + cur
            pos[1, i:i + n_tok] = h_idx + cur
            pos[2, i:i + n_tok] = w_idx + cur

            # ★ 关键：图像之后的位置基准要跳到三个分量的最大值 + 1
            #   不是 cur + n_tok。写错这一行，图后面的文本全部错位。
            cur = cur + max(int(t), int(h_m), int(w_m))
            i += n_tok
            img_idx += 1

        elif tok == VIDEO_PAD_ID and video_grid_thw is not None:
            t, h, w = video_grid_thw[vid_idx]
            h_m, w_m = h // spatial_merge_size, w // spatial_merge_size
            n_tok = int(t * h_m * w_m)
            pos[0, i:i + n_tok] = np.repeat(np.arange(t), h_m * w_m) + cur
            pos[1, i:i + n_tok] = np.tile(np.repeat(np.arange(h_m), w_m), t) + cur
            pos[2, i:i + n_tok] = np.tile(np.arange(w_m), t * h_m) + cur
            cur = cur + max(int(t), int(h_m), int(w_m))
            i += n_tok
            vid_idx += 1

        else:
            # 文本 token：三分量相同
            pos[:, i] = cur
            cur += 1
            i += 1

    mrope_delta = cur - seq_len
    return pos, mrope_delta


def build_prompt_table(
    input_ids: np.ndarray,
    vision_embeds: np.ndarray,
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    prompt table 机制（TensorRT-LLM 的 ptuning 复用）
    -------------------------------------------------
    TRT-LLM 的 LLM engine 只吃 token id，不吃 embedding。要把 ViT 输出的连续
    向量塞进去，走的是 prompt tuning 那套：

      1. 把 vision_embeds 打包成 prompt_table [n_visual, hidden]
      2. 把 input_ids 里的 image_pad 占位符替换成 **fake id**：
             fake_id = vocab_size + row_index_in_prompt_table
      3. engine 内部 embedding lookup 时，id >= vocab_size 的走 prompt_table

    注意
    ----
    - prompt_table 的 dtype 必须和 engine 的 dtype 一致（FP16）
    - hidden 维度必须等于 LLM 的 hidden_size，不等说明 merger 输出维度没对上
    - build engine 时 max_multimodal_len 必须 >= n_visual，否则运行时报错
    """
    n_visual, hidden = vision_embeds.shape
    fake_ids = input_ids.copy()

    mask = (fake_ids == IMAGE_PAD_ID) | (fake_ids == VIDEO_PAD_ID)
    n_slots = int(mask.sum())

    if n_slots != n_visual:
        raise ValueError(
            f"占位符数量({n_slots}) != 视觉 token 数({n_visual})。\n"
            f"常见原因：\n"
            f"  1. processor 的 min_pixels/max_pixels 与 ViT engine 的 "
            f"optimization profile 不匹配\n"
            f"  2. spatial_merge_size 配置错误\n"
            f"  3. 多图输入时 grid_thw 顺序和图片顺序对不上"
        )

    fake_ids[mask] = vocab_size + np.arange(n_visual, dtype=fake_ids.dtype)
    return fake_ids, vision_embeds.astype(np.float16)


def decode_step_position(
    step: int,
    prompt_len: int,
    mrope_delta: int,
) -> np.ndarray:
    """
    decode 阶段每个新 token 的位置。
    三分量相同（文本 token），值 = prompt_len + step + mrope_delta

    这里最容易忘记加 mrope_delta。忘了的现象是：prefill 输出正常，
    decode 到十几个 token 后开始重复或者语无伦次。
    """
    p = prompt_len + step + mrope_delta
    return np.array([[p], [p], [p]], dtype=np.int64)


# --------------------------------------------------------------------------- #
# 自检：跟 HF 原版对拍
# --------------------------------------------------------------------------- #

def selftest_against_hf(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    """
    跑通这个函数再往下做。不通过就说明位置编码有问题，
    后面所有 benchmark 数据都不可信。

    版本兼容说明
    ------------
    - Qwen2.5-VL 自 transformers 4.49 起原生支持，不需要 trust_remote_code；
      重构后 get_rope_index 从 ForConditionalGeneration 外层移到了内层
      Qwen2_5_VLModel 上，这里两处都探测。
    - 模型用 accelerate 的 init_empty_weights 按 config 实例化（meta device，
      不下载/不加载权重）：get_rope_index 是纯索引逻辑，不碰任何参数，
      所以自检只需要 config + processor 两个小文件。
    """
    import torch
    from transformers import AutoConfig, AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    from accelerate import init_empty_weights
    from PIL import Image

    proc = AutoProcessor.from_pretrained(model_id)
    cfg = AutoConfig.from_pretrained(model_id)
    with init_empty_weights():
        model = Qwen2_5_VLForConditionalGeneration(cfg)
    # 4.49 前后：外层 model.get_rope_index；重构后：model.model.get_rope_index
    get_rope_index = getattr(model, "get_rope_index", None)
    if get_rope_index is None:
        get_rope_index = model.model.get_rope_index

    def run_case(name, images, prompt="这是什么"):
        content = [{"type": "image"}] * len(images) + [{"type": "text", "text": prompt}]
        msgs = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if images:
            inputs = proc(text=[text], images=images, return_tensors="pt")
            grid = inputs["image_grid_thw"]
        else:
            inputs = proc(text=[text], return_tensors="pt")
            grid = None

        ref_pos, ref_delta = get_rope_index(
            inputs["input_ids"], grid,
            attention_mask=inputs["attention_mask"])

        ours, our_delta = compute_mrope_position_ids(
            inputs["input_ids"][0].numpy(),
            grid.numpy() if grid is not None else None)

        ref = ref_pos[:, 0, :].numpy()
        same = np.array_equal(ref, ours)
        delta_same = (our_delta == int(ref_delta.reshape(-1)[0].item()))
        print(f"[{name}] seq_len={ref.shape[1]}  position_ids 一致: {same}  "
              f"delta ours={our_delta} ref={int(ref_delta.reshape(-1)[0].item())} 一致: {delta_same}")
        if not same:
            bad = np.argwhere(ref != ours)
            print(f"  首个不一致位置: {bad[0]}")
            print(f"  ref ={ref[:, bad[0][1]]}")
            print(f"  ours={ours[:, bad[0][1]]}")
        assert same and delta_same, f"[{name}] M-RoPE 实现与 HF 不一致，不要继续往下做"

    # 用例1：单图 + 图后文本（最常见路径，覆盖"图后文本接 max+1"）
    run_case("单图448x448", [Image.new("RGB", (448, 448), (128, 128, 128))])
    # 用例2：双图不同尺寸（覆盖多图 t 递增与占位符顺序匹配）
    run_case("双图448+308x224", [
        Image.new("RGB", (448, 448), (100, 100, 100)),
        Image.new("RGB", (308, 224), (200, 200, 200))])
    # 用例3：纯文本（覆盖三分量退化为 1D 的兼容路径）
    run_case("纯文本", [])

    print("自检通过：3 组用例 position_ids 与 mrope_delta 全部与 HF 一致")


if __name__ == "__main__":
    selftest_against_hf()
