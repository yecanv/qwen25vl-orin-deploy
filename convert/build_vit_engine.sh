#!/usr/bin/env bash
# =============================================================================
# ViT ONNX → TensorRT engine (FP16)
# 必须在 Orin 板卡上执行，engine 绑定 SM87 架构
# =============================================================================
set -euo pipefail

ONNX=${1:-onnx/vit_fp16.onnx}
ENGINE=${2:-engines/vit_fp16.engine}
mkdir -p "$(dirname "$ENGINE")"

# ---- optimization profile ----------------------------------------------------
# 动态 shape 必须给三档。这三个数直接决定性能，按你的实际输入分布来调：
#   MIN : 最小可能输入（缩略图 / 低分辨率）
#   OPT : 主力分辨率 —— TensorRT 按这一档做 kernel autotuning，设准了最重要
#   MAX : 上限（文档类高分辨率），设太大会吃掉一大块工作空间显存
#
# 换算：n_patches = (H/14) * (W/14) * 2      # 2 是 temporal patch
#       视觉 token = n_patches / 4           # merger 2x2 合并
MIN_PATCHES=${MIN_PATCHES:-256}
OPT_PATCHES=${OPT_PATCHES:-4096}     # ≈1024 视觉 token，对应 ~896x896
MAX_PATCHES=${MAX_PATCHES:-16384}    # ≈4096 视觉 token，对应 DocVQA 那种大图

# 工作空间：Orin 是统一内存，别设太大，8GB 板子建议 1024
WORKSPACE=${WORKSPACE:-2048}

echo "==> building ViT engine"
echo "    profile: min=$MIN_PATCHES opt=$OPT_PATCHES max=$MAX_PATCHES"
echo "    workspace: ${WORKSPACE}MiB"

trtexec \
  --onnx="$ONNX" \
  --saveEngine="$ENGINE" \
  --fp16 \
  --minShapes=pixel_values:${MIN_PATCHES}x1176,grid_thw:1x3 \
  --optShapes=pixel_values:${OPT_PATCHES}x1176,grid_thw:1x3 \
  --maxShapes=pixel_values:${MAX_PATCHES}x1176,grid_thw:1x3 \
  --memPoolSize=workspace:${WORKSPACE} \
  --builderOptimizationLevel=4 \
  --useCudaGraph \
  --verbose 2>&1 | tee logs/vit_build.log

echo "==> engine: $ENGINE"
ls -lh "$ENGINE"

# 记录构建环境，面试被问"什么环境跑的"要拿得出来
{
  echo "build_time=$(date -Iseconds)"
  echo "trtexec=$(trtexec --version 2>&1 | head -1)"
  echo "jetpack=$(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
  echo "nvpmodel=$(sudo nvpmodel -q 2>/dev/null | tail -1)"
  echo "profile=min:${MIN_PATCHES} opt:${OPT_PATCHES} max:${MAX_PATCHES}"
} > "${ENGINE}.buildinfo"
cat "${ENGINE}.buildinfo"
