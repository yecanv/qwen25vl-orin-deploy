#!/usr/bin/env bash
# 一键跑全套 benchmark。在 Orin 上执行，预计 1.5-2 小时。
set -euo pipefail

mkdir -p results/raw logs

echo "==> [0/6] 准备：测试图 + 环境基线"
[[ -f assets/demo.jpg ]] || python3 assets/make_demo.py
python3 benchmark/probe_device.py

echo "==> [1/6] 确认功耗模式"
sudo nvpmodel -q
sudo jetson_clocks
echo "    等待 60s 让温度稳定…"
sleep 60

for TAG in fp16 int8sq int4awq; do
  ENGINE="engines/llm_${TAG}"
  [[ -d "$ENGINE" ]] || { echo "跳过 $TAG（engine 不存在）"; continue; }

  echo "==> [2/6] 延迟分解 ($TAG)"
  python3 benchmark/bench_latency.py --llm-engine "$ENGINE" --tag "$TAG"

  echo "==> [3/6] 并发吞吐 ($TAG)"
  python3 benchmark/bench_throughput.py --llm-engine "$ENGINE" --tag "$TAG" || true

  echo "    降温 120s（避免上一轮的热量影响下一轮，这步别省）"
  sleep 120
done

echo "==> [4/6] 量化敏感度（可在桌面卡上跑，更快）"
python3 quantize/sensitivity_analysis.py || true

echo "==> [5/6] 长稳 30min（用主力方案）"
python3 benchmark/bench_power_stability.py --duration 1800

echo "==> [6/6] 出图 + 填表"
python3 benchmark/plot_results.py --fill-template

echo ""
echo "完成。检查 results/ 下的数据表，空格子说明对应测试没跑通。"
