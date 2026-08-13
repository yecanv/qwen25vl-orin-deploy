#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase1 板端段:服务启动→health→最小推理,各段计时(板钟)"""
import json, os, subprocess, time, urllib.request
LC = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
GG = os.path.expanduser("~/models/gguf")
subprocess.run("pkill -f llama-serve[r]", shell=True); time.sleep(2)
t0 = time.time()
p = subprocess.Popen([LC, "-m", f"{GG}/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "--mmproj", f"{GG}/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf",
    "-ngl", "99", "--host", "0.0.0.0", "--port", "8080", "-c", "4096", "-np", "2"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
while True:
    try:
        urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=1); break
    except Exception: time.sleep(0.2)
t_health = time.time() - t0
t1 = time.time()
d = json.dumps({"prompt": "你好", "n_predict": 8, "temperature": 0.0}).encode()
urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8080/completion",
    data=d, headers={"Content-Type": "application/json"}), timeout=120).read()
t_infer = time.time() - t1
temps = subprocess.run("cat /sys/devices/virtual/thermal/thermal_zone*/temp", shell=True,
                       capture_output=True, text=True).stdout.split()
print(json.dumps({"start_to_health_s": round(t_health, 2),
                  "first_min_inference_s": round(t_infer, 2),
                  "thermal_mC": temps[:4]}))
