#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""长稳(30min)+并发(4x3min)客户端 → CSV + 汇总 json（板上运行，仅标准库）"""
import json, time, urllib.request, threading, os, re, statistics

BASE = "http://127.0.0.1:8080"
OUT = os.path.expanduser("~/repeat_defaultfan")
PROMPT = "请写一段关于城市清晨的描写，包含街道、行人和天气。"

def req(n_predict=128, timeout=300):
    data = json.dumps({"prompt": PROMPT, "n_predict": n_predict,
                       "temperature": 0.7, "cache_prompt": False}).encode()
    t0 = time.time()
    r = urllib.request.urlopen(urllib.request.Request(
        BASE + "/completion", data=data,
        headers={"Content-Type": "application/json"}), timeout=timeout)
    j = json.loads(r.read())
    dt = time.time() - t0
    t = j.get("timings", {})
    return dt, t.get("predicted_per_second"), t.get("predicted_n"), t.get("prompt_per_second")

# ---- 等 server 就绪 ----
for i in range(120):
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
        print("server ready", flush=True)
        break
    except Exception:
        time.sleep(5)
else:
    raise SystemExit("server 未就绪")

# ---- 阶段A：30 分钟长稳 ----
print("PHASE_A_START", time.time(), flush=True)
fa = open(f"{OUT}/longrun_seq.csv", "w")
fa.write("ts,latency_s,decode_tps,ntok,prefill_tps\n")
end = time.time() + 1800
n_req = 0
while time.time() < end:
    try:
        dt, tps, ntok, ptps = req()
        fa.write(f"{time.time():.1f},{dt:.2f},{tps:.2f},{ntok},{ptps:.1f}\n")
        fa.flush()
        n_req += 1
        if n_req % 20 == 0:
            print(f"A[{n_req}] tps={tps:.1f}", flush=True)
    except Exception as e:
        print("A err:", type(e).__name__, flush=True)
        time.sleep(3)
fa.close()
print("PHASE_A_END", time.time(), flush=True)

# ---- 阶段C：4 线程并发 3 分钟 ----
print("PHASE_C_START", time.time(), flush=True)
lock = threading.Lock()
c_rows, c_tokens = [], [0]
c_end = time.time() + 180

def worker(wid):
    while time.time() < c_end:
        try:
            dt, tps, ntok, _ = req()
            with lock:
                c_rows.append((time.time(), wid, dt, tps, ntok))
                c_tokens[0] += ntok or 0
        except Exception as e:
            with lock:
                print(f"C{wid} err:", type(e).__name__, flush=True)
            time.sleep(2)

t_c0 = time.time()
ths = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
[t.start() for t in ths]
[t.join() for t in ths]
c_dur = time.time() - t_c0
with open(f"{OUT}/longrun_conc.csv", "w") as fc:
    fc.write("ts,worker,latency_s,decode_tps,ntok\n")
    for r in c_rows:
        fc.write(",".join(str(x) for x in r) + "\n")
print("PHASE_C_END", time.time(), flush=True)

# ---- 解析 tegrastats（VDD_IN 功耗 + gpu 温度）----
power, temps = [], []
try:
    for line in open("/tmp/tegra_longrun.log", encoding="utf-8", errors="replace"):
        m = re.search(r"VDD_IN (\d+)mW", line)
        if m: power.append(int(m.group(1)))
        m = re.search(r"gpu@([\d.]+)C", line)
        if m: temps.append(float(m.group(1)))
except FileNotFoundError:
    pass

# ---- 阶段A 窗口漂移 ----
rows = []
for ln in open(f"{OUT}/longrun_seq.csv"):
    if ln.startswith("ts"): continue
    p = ln.strip().split(",")
    rows.append((float(p[0]), float(p[2])))
t0a = rows[0][0] if rows else 0
wins = {}
for ts, tps in rows:
    wins.setdefault(int((ts - t0a) // 300), []).append(tps)
win_stats = {f"win{k}_5min": {"p50": round(statistics.median(v), 2),
                              "p95": round(sorted(v)[max(0,int(len(v)*0.95)-1)], 2),
                              "n": len(v)} for k, v in sorted(wins.items())}
first = win_stats.get("win0_5min", {}).get("p50")
last = win_stats.get(f"win{max(wins)}_5min", {}).get("p50") if wins else None
gen_tokens_a = sum(1 for _ in rows) * 128
avg_p = statistics.mean(power) if power else None

summary = {
    "phase_a_requests": len(rows),
    "windows": win_stats,
    "p50_drift_pct": round((last-first)/first*100, 2) if first and last else None,
    "power_mW": {"avg": round(avg_p) if avg_p else None,
                 "max": max(power) if power else None},
    "gpu_temp_C": {"max": max(temps) if temps else None,
                   "avg": round(statistics.mean(temps),1) if temps else None},
    "energy_efficiency_tok_per_J": round(gen_tokens_a / (avg_p/1000 * 1800), 2) if avg_p else None,
    "phase_c": {"duration_s": round(c_dur,1), "total_tokens": c_tokens[0],
                "aggregate_tps": round(c_tokens[0]/c_dur, 2),
                "requests": len(c_rows)},
}
json.dump(summary, open(f"{OUT}/longrun_summary.json", "w"), ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
print("LONGRUN_ALL_DONE", flush=True)
