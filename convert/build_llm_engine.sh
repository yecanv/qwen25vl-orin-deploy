#!/usr/bin/env bash
# =============================================================================
# 量化后的 LLM checkpoint → TensorRT-LLM engine
# 必须在 Orin 上执行
# =============================================================================
set -euo pipefail

CKPT=${1:-ckpt/qwen25vl-3b-int4awq}
ENGINE=${2:-engines/llm_int4awq}

# ---- 关键参数 ----------------------------------------------------------------
# max_multimodal_len : 单次请求允许的最大视觉 token 数
#                      必须 >= build_vit_engine.sh 里 MAX_PATCHES/4
#                      设小了运行时直接报错，设大了白白吃 KV Cache 预算
MAX_MM_LEN=${MAX_MM_LEN:-4096}

# max_input_len 要把视觉 token 算进去：视觉 token + 文本 prompt
MAX_INPUT_LEN=${MAX_INPUT_LEN:-6144}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-8192}
MAX_BATCH=${MAX_BATCH:-4}

# KV Cache 显存占比。Orin 统一内存，留余量给 ViT engine 和系统
KV_FRACTION=${KV_FRACTION:-0.55}

mkdir -p "$ENGINE" logs

echo "==> building LLM engine"
trtllm-build \
  --checkpoint_dir "$CKPT" \
  --output_dir "$ENGINE" \
  --gemm_plugin auto \
  --gpt_attention_plugin auto \
  --max_batch_size "$MAX_BATCH" \
  --max_input_len "$MAX_INPUT_LEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --max_multimodal_len "$MAX_MM_LEN" \
  --use_paged_context_fmha enable \
  --kv_cache_type paged \
  --remove_input_padding enable \
  --context_fmha enable \
  2>&1 | tee logs/llm_build.log

echo "==> engine: $ENGINE"
du -sh "$ENGINE"

{
  echo "build_time=$(date -Iseconds)"
  echo "checkpoint=$CKPT"
  echo "max_multimodal_len=$MAX_MM_LEN"
  echo "max_input_len=$MAX_INPUT_LEN  max_seq_len=$MAX_SEQ_LEN"
  echo "max_batch_size=$MAX_BATCH  kv_fraction=$KV_FRACTION"
  echo "jetpack=$(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
  cat "$CKPT/quant_recipe.json" 2>/dev/null || true
} > "$ENGINE/buildinfo.txt"
cat "$ENGINE/buildinfo.txt"
