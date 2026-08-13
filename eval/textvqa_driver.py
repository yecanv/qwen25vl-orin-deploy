# -*- coding: utf-8 -*-
"""板端 TextVQA 批跑驱动:两个模型(f16 / Q4_K_M)逐题推理,预测落 jsonl
用法: python3 textvqa_driver.py  (在 ~/textvqa/ 目录,内含 images/ questions.jsonl)"""
import json, subprocess, time, os, sys

BIN = "/home/nvidia/llama.cpp/build/bin/llama-mtmd-cli"
MMPROJ = "/home/nvidia/models/gguf/mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"
MODELS = {
    "q4km": "/home/nvidia/models/gguf/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
    "f16":  "/home/nvidia/models/gguf/qwen25vl-3b-text-f16.gguf",
}
BASE = os.path.expanduser("~/textvqa")
PROMPT_SUFFIX = "\nAnswer the question using a single word or phrase."

questions = [json.loads(l) for l in open(f"{BASE}/questions.jsonl", encoding="utf-8")]
print(f"共 {len(questions)} 题", flush=True)

for tag, model in MODELS.items():
    out_path = f"{BASE}/preds_{tag}.jsonl"
    done = set()
    if os.path.exists(out_path):                       # 断点续跑
        done = {json.loads(l)["qid"] for l in open(out_path, encoding="utf-8")}
    fout = open(out_path, "a", encoding="utf-8")
    t0 = time.time()
    for i, q in enumerate(questions):
        if q["qid"] in done:
            continue
        cmd = [BIN, "-m", model, "--mmproj", MMPROJ, "-ngl", "99",
               "--image", f"{BASE}/images/{q['image']}",
               "-p", q["question"] + PROMPT_SUFFIX,
               "-n", "16", "--temp", "0"]
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=120)
            pred = r.stdout.decode("utf-8", "replace").strip().splitlines()
            pred = pred[-1].strip() if pred else ""
        except subprocess.TimeoutExpired:
            pred = "__TIMEOUT__"
        fout.write(json.dumps({"qid": q["qid"], "pred": pred},
                              ensure_ascii=False) + "\n")
        fout.flush()
        if (i + 1) % 20 == 0:
            el = time.time() - t0
            print(f"[{tag}] {i+1}/{len(questions)}  {el/60:.1f}min", flush=True)
    fout.close()
    print(f"[{tag}] 完成 → {out_path}", flush=True)

print("TEXTVQA_DRIVER_DONE", flush=True)
