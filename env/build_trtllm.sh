#!/usr/bin/env bash
# =============================================================================
# TensorRT-LLM on Jetson Orin (ARM64 / SM87)
#
# 老实说：这一步是整个项目最耗时、最容易失败的地方，预留 3-6 小时。
# 提供两条路，先试 A，A 不行再走 B。
# =============================================================================
set -euo pipefail

echo "路线 A：jetson-containers 预编译镜像（推荐，省 4 小时）"
echo "路线 B：源码编译（版本不匹配时的兜底）"
echo ""
read -rp "选择 [A/B]: " ROUTE

if [[ "${ROUTE^^}" == "A" ]]; then
  # ---------------------------------------------------------------------------
  # 路线 A：dusty-nv/jetson-containers
  # 优点：省编译时间；缺点：镜像 ~15GB，且 TRT-LLM 版本跟随镜像，不一定最新
  # ---------------------------------------------------------------------------
  echo "==> clone jetson-containers"
  [[ -d jetson-containers ]] || \
    git clone --depth 1 https://github.com/dusty-nv/jetson-containers
  cd jetson-containers && bash install.sh

  echo "==> 拉取 TRT-LLM 镜像（按你的 JetPack 版本自动匹配）"
  jetson-containers run $(autotag tensorrt_llm)
  echo ""
  echo "进容器后验证： python3 -c 'import tensorrt_llm; print(tensorrt_llm.__version__)'"

else
  # ---------------------------------------------------------------------------
  # 路线 B：源码编译
  # ---------------------------------------------------------------------------
  echo "==> 源码编译 TensorRT-LLM for SM87"
  echo ""
  echo "!! 编译前确认（版本对不上就是浪费 4 小时）："
  echo "   - JetPack 版本： $(cat /etc/nv_tegra_release 2>/dev/null | head -1)"
  echo "   - TensorRT 版本：$(dpkg -l | grep -m1 libnvinfer | awk '{print $3}')"
  echo "   - 到 TRT-LLM release notes 查哪个 tag 支持你这个 TRT 版本"
  echo "   - Qwen2-VL 支持从 v0.12 起，Qwen2.5-VL 需要更新的 tag，务必先查"
  echo ""
  read -rp "   要 checkout 的 tag (如 v0.15.0): " TAG

  [[ -d TensorRT-LLM ]] || \
    git clone https://github.com/NVIDIA/TensorRT-LLM.git
  cd TensorRT-LLM
  git checkout "$TAG"
  git submodule update --init --recursive
  git lfs pull

  echo "==> 开始编译（SM87，只编这一个架构，能省一半时间）"
  echo "   内存不足会在链接阶段 OOM，确保 swap 已开（env/jetson_setup.sh）"
  python3 scripts/build_wheel.py \
      --clean \
      --cuda_architectures "87-real" \
      --build_type Release \
      --job_count 4 \
      --benchmarks 2>&1 | tee ../logs/trtllm_build.log

  pip3 install build/tensorrt_llm-*.whl
fi

echo ""
echo "==> 验证"
python3 - <<'PY'
try:
    import tensorrt_llm, tensorrt, torch
    print("tensorrt_llm", tensorrt_llm.__version__)
    print("tensorrt    ", tensorrt.__version__)
    print("torch       ", torch.__version__, "cuda", torch.version.cuda)
    print("device      ", torch.cuda.get_device_name(0))
    p = torch.cuda.get_device_properties(0)
    print("sm          ", f"{p.major}{p.minor}", "(Orin 应为 87)")
except Exception as e:
    print("FAILED:", e)
PY
