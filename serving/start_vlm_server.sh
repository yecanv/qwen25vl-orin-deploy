#!/bin/sh
# 起 VLM HTTP 服务(OpenAI 兼容 API):Q4_K_M + mmproj,监听局域网
LC=/home/nvidia/llama.cpp/build/bin
GG=/home/nvidia/models/gguf
LOG=/home/nvidia/vlm_server.log

pkill -f "llama-server" 2>/dev/null
sleep 2
nohup $LC/llama-server \
  -m $GG/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
  --mmproj $GG/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf \
  -ngl 99 --host 0.0.0.0 --port 8080 \
  -c 4096 -np 2 --metrics \
  > $LOG 2>&1 < /dev/null &
echo "SERVER_LAUNCHED pid=$!"
sleep 25
tail -5 $LOG
curl -s -m 5 http://127.0.0.1:8080/health && echo " <- health"
