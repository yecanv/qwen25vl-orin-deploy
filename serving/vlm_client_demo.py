# -*- coding: utf-8 -*-
"""桌面 → 板端 VLM 服务:OpenAI 兼容图文请求 + serving 口径指标"""
import io, sys, os, json, base64, time, urllib.request
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

URL = "http://192.168.1.2:8080/v1/chat/completions"
SUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textvqa_subset", "images")
img_file = sorted(os.listdir(SUB))[0]
b64 = base64.b64encode(open(os.path.join(SUB, img_file), "rb").read()).decode()

def ask(prompt, stream=False):
    payload = {
        "model": "qwen2.5-vl-3b",
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            {"type": "text", "text": prompt}]}],
        "max_tokens": 128, "temperature": 0.2, "stream": stream,
    }
    req = urllib.request.Request(URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=180) as r:
        body = json.loads(r.read().decode())
    dt = time.perf_counter() - t0
    return body, dt

print(f"图片: {img_file}\n请求中(第一次含预热)…")
body, dt = ask("详细描述这张图片的内容。")
msg = body["choices"][0]["message"]["content"]
u = body.get("usage", {})
print("-" * 60)
print("模型回答:", msg[:400])
print("-" * 60)
print(f"整轮耗时 {dt:.2f}s  prompt_tokens={u.get('prompt_tokens')} "
      f"completion_tokens={u.get('completion_tokens')}")
if u.get("completion_tokens"):
    print(f"serving 口径整体速率 ≈ {u['completion_tokens']/dt:.1f} tok/s(含视觉编码与网络)")

print("\n第二次(热态,换个问题)…")
body2, dt2 = ask("图中有文字吗?如果有,原样写出来。")
u2 = body2.get("usage", {})
print("回答:", body2["choices"][0]["message"]["content"][:200])
print(f"整轮耗时 {dt2:.2f}s  completion_tokens={u2.get('completion_tokens')}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "serving_demo.json")
json.dump({"image": img_file,
           "round1": {"latency_s": round(dt, 2), "usage": u, "answer": msg},
           "round2": {"latency_s": round(dt2, 2), "usage": u2,
                      "answer": body2["choices"][0]["message"]["content"]}},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("→", out)
