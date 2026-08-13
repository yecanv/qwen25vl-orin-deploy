#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLM 的 LLM 主干量化（ViT 保 FP16）
=================================

核心决策：**只量化 LLM 主干，视觉编码器保持 FP16。**

理由（面试必问，务必能自己讲清楚）：

1. ViT 对量化敏感得多。
   - ViT 的 LayerNorm 后激活存在强 outlier，且 outlier 出现在**不固定的
     channel** 上（不像 LLM 的 outlier channel 相对稳定），per-channel scale
     压不住。
   - ViT 只有 ~0.6B 参数（Qwen2.5-VL-3B 里视觉部分约 670M），量化它省下的
     显存有限，但精度损失可能让 OCR/细粒度识别直接崩掉。

2. 收益分配不对等。
   - LLM 主干占参数量 ~75%，且 decode 阶段是 **memory-bound**，权重量化直接
     转化为吞吐提升。
   - ViT 只在 prefill 跑一次，是 **compute-bound**，量化它对 TTFT 的改善远小于
     对精度的伤害。

3. 这个取舍本身是可验证的。
   `quantize/sensitivity_analysis.py` 会跑出逐模块的量化敏感度曲线，
   用数据说话，而不是"业界都这么做"。

支持的量化格式：
  int8_sq    SmoothQuant W8A8，兼容性最好，Orin 上 INT8 Tensor Core 吃满
  int4_awq   AWQ W4A16，压缩比最高，8GB 板子的唯一选择
  w4a8_awq   权重 INT4 + 激活 INT8，理论最优，但 TRT-LLM 在 SM87 上支持要验
  fp8        Orin 是 SM87（Ampere），**不支持 FP8**，别浪费时间试

用法：
  python convert/quantize_llm.py \
      --model Qwen/Qwen2.5-VL-3B-Instruct \
      --qformat int4_awq \
      --calib calib/data/vl_calib_512.pt \
      --out ckpt/qwen25vl-3b-int4awq
"""

import argparse
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch


# --------------------------------------------------------------------------- #
# 需要排除在量化之外的模块
# --------------------------------------------------------------------------- #

VISION_MODULE_PATTERNS = [
    "visual",            # Qwen2-VL / 2.5-VL 视觉塔的顶层前缀
    "vision_tower",
    "merger",            # patch merger（2x2 → 1 token），参数少但精度关键
    "vision_model",
]

# LLM 内部也有不该量化的部分
LLM_SKIP_PATTERNS = [
    "lm_head",           # 输出层量化会直接放大 logits 误差
    "embed_tokens",      # 查表操作，量化无收益
    "*.norm",            # RMSNorm 权重是逐通道缩放，量化后误差沿层累积
    "*.input_layernorm",
    "*.post_attention_layernorm",
]


def build_quant_config(qformat: str, group_size: int = 128) -> Dict[str, Any]:
    """构造 ModelOpt 量化配置，并把视觉塔整体排除。"""
    import modelopt.torch.quantization as mtq

    base = {
        "int8_sq":   mtq.INT8_SMOOTHQUANT_CFG,
        "int4_awq":  mtq.INT4_AWQ_CFG,
        "w4a8_awq":  mtq.W4A8_AWQ_BETA_CFG,
    }.get(qformat)

    if base is None:
        raise ValueError(
            f"不支持的 qformat: {qformat}\n"
            f"注意：Orin 是 SM87 (Ampere)，硬件不支持 FP8，"
            f"想用 FP8 得上 Ada/Hopper。"
        )

    cfg = {k: v for k, v in base.items()}
    quant_cfg = dict(cfg["quant_cfg"])

    # 关键一步：把视觉塔和敏感层全部设为不量化
    for pat in VISION_MODULE_PATTERNS:
        quant_cfg[f"*{pat}*"] = {"enable": False}
    for pat in LLM_SKIP_PATTERNS:
        quant_cfg[pat] = {"enable": False}

    if qformat in ("int4_awq", "w4a8_awq"):
        # group_size 影响精度/速度：128 是通用甜点，64 精度更好但 kernel 更慢
        for k, v in quant_cfg.items():
            if isinstance(v, dict) and v.get("num_bits") == 4:
                v["block_sizes"] = {-1: group_size}

    cfg["quant_cfg"] = quant_cfg
    return cfg


# --------------------------------------------------------------------------- #
# 校准 loop
# --------------------------------------------------------------------------- #

def make_calib_loop(model, samples: List[Dict[str, Any]], device: str,
                    max_batch: int = 1):
    """
    ModelOpt 要求传入一个 forward_loop(model)，内部把校准样本喂一遍。

    这里必须走**完整的多模态前向**（pixel_values 一起传进去），
    这样视觉 token 的激活才会被统计到。只传 input_ids 是错的，
    见 calib/build_vl_calib.py 顶部的说明。
    """
    def forward_loop(m):
        m.eval()
        t0 = time.time()
        with torch.no_grad():
            for i, s in enumerate(samples):
                batch = {
                    "input_ids": s["input_ids"].unsqueeze(0).to(device),
                    "attention_mask": s["attention_mask"].unsqueeze(0).to(device),
                    "pixel_values": s["pixel_values"].to(device, dtype=torch.float16),
                    "image_grid_thw": s["image_grid_thw"].to(device),
                }
                try:
                    m(**batch)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  [warn] 样本 {i} OOM 跳过 "
                          f"(seq_len={batch['input_ids'].shape[1]})，"
                          f"建议降低 build_vl_calib.py 的 --max-pixels")
                    continue

                if (i + 1) % 32 == 0:
                    el = time.time() - t0
                    print(f"  校准进度 {i+1}/{len(samples)}  "
                          f"{el:.0f}s  ({el/(i+1):.2f}s/sample)")
        print(f"  校准完成，耗时 {time.time() - t0:.0f}s")

    return forward_loop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--qformat", default="int4_awq",
                    choices=["int8_sq", "int4_awq", "w4a8_awq"])
    ap.add_argument("--calib", default="calib/data/vl_calib_512.pt")
    ap.add_argument("--out", default="ckpt/qwen25vl-3b-int4awq")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--calib-size", type=int, default=0,
                    help="0 = 用全部；用于 128/256/512 消融实验")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print("=" * 70)
    print("注意：量化在**桌面卡**上做（需要 FP16 全模型 + 激活统计，")
    print("      3B 模型峰值约 14-18GB）。Orin 板子内存不够，跑不动。")
    print("      产出的 quantized checkpoint 再拷到 Orin 上 build engine。")
    print("=" * 70)

    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    from transformers import AutoModelForVision2Seq

    print(f"\n[1/4] 加载模型 {args.model}")
    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map=args.device, trust_remote_code=True,
    )

    print(f"[2/4] 加载校准集 {args.calib}")
    pack = torch.load(args.calib, weights_only=False)
    samples = pack["samples"]
    if args.calib_size > 0:
        samples = samples[:args.calib_size]
    print(f"      样本数 {len(samples)}，来源分布 {pack['stats']['by_source']}")
    print(f"      序列长度 p50={pack['stats']['seq_len_p50']} "
          f"p95={pack['stats']['seq_len_p95']}")

    print(f"[3/4] 量化 qformat={args.qformat} group_size={args.group_size}")
    cfg = build_quant_config(args.qformat, args.group_size)
    n_excluded = sum(1 for v in cfg["quant_cfg"].values()
                     if isinstance(v, dict) and v.get("enable") is False)
    print(f"      已排除 {n_excluded} 类模块（视觉塔 + norm + lm_head）")

    loop = make_calib_loop(model, samples, args.device)
    model = mtq.quantize(model, cfg, loop)

    print(f"[4/4] 导出 TensorRT-LLM checkpoint → {args.out}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        export_tensorrt_llm_checkpoint(
            model.language_model if hasattr(model, "language_model") else model,
            "qwen",
            torch.float16,
            export_dir=str(out),
            inference_tensor_parallel=1,
            inference_pipeline_parallel=1,
        )

    # 记录量化配置，供复现与面试说明
    with open(out / "quant_recipe.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "qformat": args.qformat,
            "group_size": args.group_size,
            "calib_samples": len(samples),
            "calib_source": pack["meta"]["sources"],
            "calib_stats": pack["stats"],
            "excluded_modules": VISION_MODULE_PATTERNS + LLM_SKIP_PATTERNS,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n完成。下一步在 Orin 上执行 convert/build_llm_engine.sh")
    print(f"提醒：engine 必须在目标板卡上 build，桌面卡编的 engine 加载不了。")


if __name__ == "__main__":
    main()
