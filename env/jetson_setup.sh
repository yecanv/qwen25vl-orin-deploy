#!/usr/bin/env bash
# =============================================================================
# Jetson Orin 环境准备
# 在板子上执行： sudo bash env/jetson_setup.sh
# =============================================================================
set -euo pipefail

echo "==> 1. 检测板卡"
MODEL=$(tr -d '\0' < /proc/device-tree/model)
echo "    $MODEL"
if [[ "$MODEL" != *Orin* ]]; then
  echo "    [!] 不是 Orin 板卡，本脚本不适用"; exit 1
fi

MEM_MB=$(free -m | awk '/Mem:/{print $2}')
echo "    内存 ${MEM_MB} MB"

echo "==> 2. 功耗模式设为 MAXN"
# Orin Nano Super / NX Super: 模式 0 通常是 MAXN
sudo nvpmodel -m 0
sudo jetson_clocks           # 锁定最高频率，消除 DVFS 抖动
sudo nvpmodel -q

echo "==> 3. 关闭图形界面（省 600MB~1GB 内存，8GB 板子必做）"
if systemctl is-active --quiet gdm3 || systemctl is-active --quiet lightdm; then
  read -rp "    检测到桌面环境，关闭？(y/N) " yn
  if [[ "$yn" == "y" ]]; then
    sudo systemctl set-default multi-user.target
    echo "    已设为命令行启动，重启后生效"
  fi
fi

echo "==> 4. 配置 swap"
# 为什么需要：trtllm-build 峰值内存远超运行时。8GB 板子不开 swap 必 OOM。
# 注意 swap 只能救 build，不能救运行时——运行时走 swap 会慢到不可用。
SWAP_GB=16
if [[ $MEM_MB -gt 12000 ]]; then SWAP_GB=8; fi
CUR_SWAP=$(free -g | awk '/Swap:/{print $2}')
if [[ ${CUR_SWAP:-0} -lt $SWAP_GB ]]; then
  echo "    创建 ${SWAP_GB}G swap 文件"
  sudo fallocate -l ${SWAP_GB}G /swapfile || \
    sudo dd if=/dev/zero of=/swapfile bs=1M count=$((SWAP_GB*1024))
  sudo chmod 600 /swapfile
  sudo mkswap /swapfile
  sudo swapon /swapfile
  grep -q '/swapfile' /etc/fstab || \
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi
free -h

echo "==> 5. zram 关闭（Jetson 默认开 zram，会和真 swap 抢 CPU）"
sudo systemctl disable nvzramconfig 2>/dev/null || true

echo "==> 6. 依赖"
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev cmake ninja-build \
    libopenblas-dev libopenmpi-dev git-lfs

echo "==> 7. 检查 CUDA / TensorRT"
nvcc --version | grep release || echo "    [!] nvcc 未找到，检查 PATH: /usr/local/cuda/bin"
dpkg -l | grep -m1 libnvinfer || echo "    [!] TensorRT 未安装"

echo ""
echo "==> 完成。下一步："
echo "    python3 benchmark/probe_device.py     # 记录环境基线"
echo "    bash env/build_trtllm.sh              # 编译 TRT-LLM（耗时长）"
echo ""
echo "提醒：jetson_clocks 重启后失效，每次重启后要重新执行。"
echo "     benchmark 前务必确认 nvpmodel -q 显示 MAXN，否则数据不可比。"
