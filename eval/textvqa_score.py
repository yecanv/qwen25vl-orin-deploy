# -*- coding: utf-8 -*-
"""拉回 TextVQA 预测并按官方 VQA 规则评分"""
import io, sys, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
import paramiko

SCRATCH = os.path.dirname(os.path.abspath(__file__))
SUB = os.path.join(SCRATCH, "textvqa_subset")

cli = paramiko.SSHClient()
cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
cli.connect("192.168.1.2", username="nvidia", timeout=20)
sftp = cli.open_sftp()
for f in ["preds_q4km.jsonl", "preds_f16.jsonl", "driver.log"]:
    sftp.get(f"/home/nvidia/textvqa/{f}", os.path.join(SUB, f))
    print("拉回", f)
sftp.close(); cli.close()

# ---- VQA 标准归一化 ----
CONTRACTIONS = {"dont": "don't", "isnt": "isn't", "cant": "can't"}
ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")
DIGIT_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
             "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
             "nine": "9", "ten": "10"}

def norm(s):
    s = s.lower().strip()
    s = PUNCT.sub(" ", s)
    toks = [DIGIT_MAP.get(t, t) for t in s.split() if t not in ARTICLES]
    return " ".join(toks).strip()

def vqa_acc(pred, answers):
    """官方规则:min(#匹配人数/3, 1)"""
    p = norm(pred)
    if not p:
        return 0.0
    hits = sum(1 for a in answers if norm(a) == p)
    return min(hits / 3.0, 1.0)

questions = {json.loads(l)["qid"]: json.loads(l)
             for l in open(os.path.join(SUB, "questions.jsonl"), encoding="utf-8")}

report = {}
for tag in ["f16", "q4km"]:
    preds = [json.loads(l) for l in
             open(os.path.join(SUB, f"preds_{tag}.jsonl"), encoding="utf-8")]
    scores, empties, timeouts = [], 0, 0
    detail = {}
    for p in preds:
        q = questions.get(p["qid"])
        if q is None:
            continue
        pred = p["pred"]
        if pred == "__TIMEOUT__":
            timeouts += 1; scores.append(0.0); continue
        if not pred.strip():
            empties += 1
        s = vqa_acc(pred, q["answers"])
        scores.append(s)
        detail[p["qid"]] = s
    acc = sum(scores) / len(scores) * 100
    report[tag] = {"n": len(scores), "acc": round(acc, 2),
                   "empty": empties, "timeout": timeouts, "detail": detail}
    print(f"{tag:5s}  n={len(scores)}  VQA-acc={acc:.2f}%  空答={empties} 超时={timeouts}")

# 逐题对比
d16, dq4 = report["f16"]["detail"], report["q4km"]["detail"]
both = set(d16) & set(dq4)
same = sum(1 for q in both if abs(d16[q] - dq4[q]) < 1e-9)
f16_win = sum(1 for q in both if d16[q] > dq4[q])
q4_win = sum(1 for q in both if dq4[q] > d16[q])
delta = report["q4km"]["acc"] - report["f16"]["acc"]
print(f"\n逐题:相同 {same} / f16胜 {f16_win} / q4km胜 {q4_win}  (共 {len(both)})")
print(f"量化代价:{delta:+.2f} 个百分点  (相对 {delta/report['f16']['acc']*100:+.1f}%)")

out = os.path.join(SUB, "textvqa_result.json")
summary = {k: {kk: vv for kk, vv in v.items() if kk != "detail"}
           for k, v in report.items()}
summary["comparison"] = {"n_both": len(both), "same": same,
                         "f16_win": f16_win, "q4km_win": q4_win,
                         "delta_points": round(delta, 2)}
summary["protocol"] = ("TextVQA validation 200 题随机采样(seed42),llama.cpp mtmd-cli,"
                       "temp=0 贪心,n=16,prompt 后缀 'Answer the question using a "
                       "single word or phrase.',官方 VQA 规则 min(hits/3,1)")
json.dump(summary, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
print("→", out)

# 打印几个分歧样例
print("\n== 分歧样例(f16 对/q4km 错) ==")
shown = 0
for q in both:
    if d16[q] > dq4[q] and shown < 5:
        qq = questions[q]
        pf = next(p["pred"] for p in
                  [json.loads(l) for l in open(os.path.join(SUB, "preds_f16.jsonl"), encoding="utf-8")]
                  if p["qid"] == q)
        pq = next(p["pred"] for p in
                  [json.loads(l) for l in open(os.path.join(SUB, "preds_q4km.jsonl"), encoding="utf-8")]
                  if p["qid"] == q)
        print(f"Q: {qq['question'][:60]}")
        print(f"   参考: {qq['answers'][:3]}")
        print(f"   f16 : {pf[:60]}")
        print(f"   q4km: {pq[:60]}")
        shown += 1
