#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 2+4:内存六点法 + idle 功耗 + 模型切换四指标(板端运行,全程板钟)
输出 ~/phase2_smaps_switch.json"""
import json, os, re, subprocess, time, urllib.request

OUT = os.path.expanduser("~/phase2_smaps_switch.json")
LC = os.path.expanduser("~/llama.cpp/build/bin/llama-server")
GG = os.path.expanduser("~/models/gguf")
rec = {"clock": "board", "stages": {}}

def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout

def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        d[k] = int(v.strip().split()[0])
    return {"MemAvailable_MB": d["MemAvailable"] // 1024,
            "MemFree_MB": d["MemFree"] // 1024, "SwapUsed_MB": (d["SwapTotal"] - d["SwapFree"]) // 1024}

def smaps(pid):
    try:
        t = open(f"/proc/{pid}/smaps_rollup").read()
        rss = int(re.search(r"Rss:\s+(\d+)", t).group(1)) // 1024
        pss = int(re.search(r"Pss:\s+(\d+)", t).group(1)) // 1024
        return {"Rss_MB": rss, "Pss_MB": pss}
    except Exception as e:
        return {"err": str(e)}

def power_mw(n=6):
    out = sh(f"timeout {n} tegrastats --interval 500 | head -{n*2}")
    vals = [int(m.group(1)) for m in re.finditer(r"VDD_IN (\d+)mW", out)]
    return {"VDD_IN_mW_avg": sum(vals)//len(vals) if vals else None, "samples": len(vals)}

def snap(name, pid=None):
    s = {"meminfo": meminfo(), "power": power_mw()}
    if pid: s["server_smaps"] = smaps(pid)
    rec["stages"][name] = s
    print(name, json.dumps(s), flush=True)

def start_server(model, port, np_, ctx, mmproj=None):
    cmd = [LC, "-m", f"{GG}/{model}", "-ngl", "99", "--host", "0.0.0.0",
           "--port", str(port), "-c", str(ctx), "-np", str(np_)]
    if mmproj: cmd += ["--mmproj", f"{GG}/{mmproj}"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 180:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
            return p, time.time() - t0
        except Exception:
            time.sleep(0.3)
    raise SystemExit("server 未就绪")

def req(port, prompt, n):
    d = json.dumps({"prompt": prompt, "n_predict": n, "temperature": 0.7,
                    "cache_prompt": False}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=d,
        headers={"Content-Type": "application/json"}), timeout=600)
    j = json.loads(r.read())
    return time.time() - t0, j.get("timings", {})

# ---- S0 基线(无服务) ----
sh("pkill -f llama-serve[r]; sleep 3")
time.sleep(5)
snap("S0_baseline_no_server")

# ---- S1 加载后空闲(Q4+mmproj,-np 8 -c 16384 与并发基准同配置) ----
p1, load_s = start_server("Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf", 8080, 8, 16384,
                          "mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf")
rec["q4_load_to_health_s"] = round(load_s, 2)
time.sleep(3)
snap("S1_loaded_idle", p1.pid)

# ---- S2 首次请求后 ----
dt, tm = req(8080, "请写一句话描述清晨。", 16)
rec["first_req_s"] = round(dt, 2)
snap("S2_after_first_request", p1.pid)

# ---- S3 长输入(约 1800 token,本配置每槽上限内) ----
long_p = "清晨的城市街道上,行人匆匆走过。" * 170 + "总结上文。"
dt, tm = req(8080, long_p, 32)
rec["long_input"] = {"latency_s": round(dt, 2), "prompt_n": tm.get("prompt_n")}
snap("S3_after_long_input", p1.pid)

# ---- S4 8 路并发峰值 ----
import threading
end = time.time() + 45
def worker():
    while time.time() < end:
        try: req(8080, "描述城市清晨,细节丰富一些。", 96)
        except Exception: pass
ths = [threading.Thread(target=worker) for _ in range(8)]
[t.start() for t in ths]
time.sleep(25)
snap("S4_during_8way_concurrent", p1.pid)
[t.join() for t in ths]

# ---- S5 模型切换:f16 与 Q4 短暂共存 → 四指标 ----
t0 = time.time()
p2, load2 = start_server("qwen25vl-3b-full-f16.gguf", 8081, 2, 4096)
snap("S5_switch_overlap_peak", p1.pid)          # 共存峰值时刻
rec["switch"] = {"f16_load_to_health_s": round(load2, 2)}
t0 = time.time()
p1.terminate(); p1.wait(timeout=30)
time.sleep(2)
rec["switch"]["q4_unload_s"] = round(time.time() - t0, 2)
snap("S6_after_q4_unloaded", p2.pid)
dt, _ = req(8081, "新模型首个请求。", 16)
rec["switch"]["f16_first_req_s"] = round(dt, 2)
p2.terminate(); p2.wait(timeout=30)
snap("S7_all_stopped")

json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("PHASE2_4_DONE", flush=True)
