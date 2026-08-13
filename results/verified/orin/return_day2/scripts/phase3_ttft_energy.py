#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 3:VLM TTFT ×50 分位数 + 每请求焦耳积分(板端运行,统一板钟)
前置:llama-server(mmproj)已在 8080 就绪
产物:~/phase3_ttft_energy.json + ~/tegrastats_phase3.log(带时间戳,10Hz)"""
import json, os, re, subprocess, time, base64, urllib.request, statistics as st

OUT = os.path.expanduser("~/phase3_ttft_energy.json")
PLOG = os.path.expanduser("~/tegrastats_phase3.log")
N_REQ = 50

# ---- 合成 10 张不同的 896² 图(避开视觉故障区+防 prompt cache) ----
from PIL import Image, ImageDraw
imgs = []
os.makedirs(os.path.expanduser("~/synimgs"), exist_ok=True)
palette = ["red", "green", "blue", "orange", "purple", "cyan", "magenta", "yellow", "gray", "brown"]
for i in range(50):
    im = Image.new("RGB", (896, 896), "white")
    d = ImageDraw.Draw(im)
    d.ellipse([50+8*i, 50+4*i, 350+8*i, 350+4*i], fill=palette[i % 10])
    d.rectangle([500, 500-20*i, 800, 800-20*i], fill=palette[(i+3) % 10])
    p = os.path.expanduser(f"~/synimgs/s{i}.png"); im.save(p); imgs.append(p)

# ---- idle 功耗 60s ----
def power_avg(sec):
    out = subprocess.run(f"timeout {sec} tegrastats --interval 500 | grep -o 'VDD_IN [0-9]*mW'",
                         shell=True, capture_output=True, text=True).stdout
    vals = [int(x) for x in re.findall(r"(\d+)mW", out)]
    return sum(vals)/len(vals) if vals else None
idle_mw = power_avg(60)
print(f"idle VDD_IN {idle_mw:.0f} mW", flush=True)

# ---- 起 10Hz 带时间戳功耗采集 ----
plog = open(PLOG, "w")
pproc = subprocess.Popen(
    "tegrastats --interval 100 | while read l; do echo \"$(date +%s.%N) $l\"; done",
    shell=True, stdout=plog, stderr=subprocess.DEVNULL)

def one_req(img):
    b64 = base64.b64encode(open(img, "rb").read()).decode()
    payload = {"model": "q", "max_tokens": 48, "temperature": 0.0, "stream": True,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
            {"type": "text", "text": "描述这张图片。"}]}]}
    t0 = time.time(); tfirst = None; ntok = 0
    r = urllib.request.urlopen(urllib.request.Request(
        "http://127.0.0.1:8080/v1/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}), timeout=300)
    for raw in r:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            delta = json.loads(line[6:])["choices"][0].get("delta", {})
            if delta.get("content"):
                ntok += 1
                if tfirst is None: tfirst = time.time()
        except Exception: pass
    return t0, tfirst, time.time(), ntok

rows = []
for k in range(N_REQ):
    t0, tf, t1, ntok = one_req(imgs[k])
    rows.append({"t0": t0, "tfirst": tf, "t1": t1, "ttft": (tf - t0) if tf else None,
                 "e2e": t1 - t0, "ntok": ntok})
    if (k+1) % 10 == 0: print(f"[{k+1}/{N_REQ}] ttft={rows[-1]['ttft']:.2f}s", flush=True)
time.sleep(2)
pproc.terminate(); subprocess.run("pkill -f tegrastats", shell=True); plog.close()

# ---- 焦耳积分:对每请求窗口积分 VDD_IN ----
samples = []
for l in open(PLOG):
    m = re.match(r"([\d.]+) .*VDD_IN (\d+)mW", l)
    if m: samples.append((float(m.group(1)), int(m.group(2))))
def energy(t0, t1):
    pts = [(t, p) for t, p in samples if t0 <= t <= t1]
    if len(pts) < 2: return None
    e = 0.0
    for (ta, pa), (tb, pb) in zip(pts, pts[1:]):
        e += (pa + pb) / 2 * (tb - ta) / 1000.0     # mW→W 积分成 J
    return e

ttfts = [r["ttft"] for r in rows if r["ttft"]]
ens = [(energy(r["t0"], r["t1"]), r["ntok"]) for r in rows]
ens = [(e, n) for e, n in ens if e and n]
jreq = [e for e, n in ens]
jtok = [e/n for e, n in ens]
def q(xs, p):
    xs = sorted(xs); return xs[max(0, min(len(xs)-1, round(p/100*len(xs))-1))]
rec = {
  "n_requests": len(rows), "images": "10 张互异合成 896²(避开视觉故障区,轮换防缓存)",
  "idle_power_mW": round(idle_mw) if idle_mw else None,
  "ttft_s": {"P50": round(q(ttfts,50),2), "P95": round(q(ttfts,95),2),
              "P99": round(q(ttfts,99),2), "max": round(max(ttfts),2), "n": len(ttfts)},
  "J_per_request": {"P50": round(q(jreq,50),1), "mean": round(st.mean(jreq),1), "n": len(jreq)},
  "J_per_token_full_request": {"P50": round(q(jtok,50),2), "mean": round(st.mean(jtok),2)},
  "口径": "完整 VLM 请求(视觉编码+prefill+decode)对 VDD_IN 10Hz 梯形积分,板钟统一;J/token 分母为输出 token 数",
}
json.dump(rec, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print(json.dumps(rec, ensure_ascii=False), flush=True)
print("PHASE3_DONE", flush=True)
