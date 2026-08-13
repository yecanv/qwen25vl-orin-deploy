#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""冷/热启动测量（server法：启动→/health 就绪计时，规避 llama-cli 退出挂死）"""
import subprocess, time, urllib.request, os, signal

BIN = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
GGUF = os.path.expanduser("~/models/gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf")
OUT = os.path.expanduser("~/work/qwen25vl-orin-deploy/results/raw/cold_start.txt")

def kill_server():
    subprocess.run(["pkill", "-9", "-f", "llama-server"], capture_output=True)
    time.sleep(2)

def measure(tag):
    kill_server()
    t0 = time.time()
    p = subprocess.Popen([BIN, "-m", GGUF, "-ngl", "99", "--port", "8080"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         stdin=subprocess.DEVNULL)
    ready = None
    while time.time() - t0 < 300:
        try:
            urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=2)
            ready = time.time() - t0
            break
        except Exception:
            time.sleep(0.5)
    print(f"{tag}: {'%.2f s' % ready if ready else 'TIMEOUT'}", flush=True)
    return ready

subprocess.run(["sudo", "-S", "bash", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
               input=b"nvidia\n", capture_output=True)
cold = measure("冷启动(drop caches后, 模型+库全从NVMe读)")
kill_server()
warm = measure("热启动(page cache 命中)")
kill_server()

with open(OUT, "w") as f:
    f.write(f"server法: 启动→/health 就绪\n")
    f.write(f"cold(drop_caches): {cold:.2f}s\n" if cold else "cold: TIMEOUT\n")
    f.write(f"warm(cache命中):   {warm:.2f}s\n" if warm else "warm: TIMEOUT\n")
print("COLD_SERVER_DONE", flush=True)
