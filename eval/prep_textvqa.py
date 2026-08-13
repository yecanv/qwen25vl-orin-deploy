# -*- coding: utf-8 -*-
"""TextVQA 子集准备:parquet -> 200 题采样 -> images/ + questions.jsonl"""
import io, sys, os, json, random
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import pandas as pd

SCRATCH = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRATCH, "textvqa_subset")
os.makedirs(os.path.join(OUT, "images"), exist_ok=True)

df = pd.read_parquet(os.path.join(SCRATCH, "textvqa_val0.parquet"))
print("列:", list(df.columns), " 总行数:", len(df))

random.seed(42)
idx = sorted(random.sample(range(len(df)), 200))
rows = []
for i in idx:
    r = df.iloc[i]
    img = r["image"]          # HF datasets 惯例: dict{bytes, path}
    img_bytes = img["bytes"] if isinstance(img, dict) else img
    fn = f"{r['question_id']}.jpg"
    with open(os.path.join(OUT, "images", fn), "wb") as f:
        f.write(img_bytes)
    answers = list(r["answers"]) if r["answers"] is not None else []
    rows.append({"qid": int(r["question_id"]), "question": str(r["question"]),
                 "answers": [str(a) for a in answers], "image": fn})

with open(os.path.join(OUT, "questions.jsonl"), "w", encoding="utf-8") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

total_mb = sum(os.path.getsize(os.path.join(OUT, "images", x))
               for x in os.listdir(os.path.join(OUT, "images"))) / 1e6
print(f"采样 {len(rows)} 题,图片合计 {total_mb:.1f}MB → {OUT}")
print("样例:", json.dumps(rows[0], ensure_ascii=False)[:200])
