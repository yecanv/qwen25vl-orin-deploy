# -*- coding: utf-8 -*-
"""TextVQA 结果方法学检查:①统计显著性(McNemar 精确检验)②截断污染面"""
import io, sys, os, json, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
from math import comb

SUB = os.path.join(os.path.dirname(os.path.abspath(__file__)), "textvqa_subset")

ARTICLES = {"a", "an", "the"}
PUNCT = re.compile(r"[;/\[\]\"{}()=+\\_\-><@`,?!.']")
DIGIT_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2", "three": "3",
             "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8",
             "nine": "9", "ten": "10"}

def norm(s):
    s = PUNCT.sub(" ", s.lower().strip())
    return " ".join(DIGIT_MAP.get(t, t) for t in s.split() if t not in ARTICLES).strip()

def vqa_acc(pred, answers):
    p = norm(pred)
    if not p: return 0.0
    return min(sum(1 for a in answers if norm(a) == p) / 3.0, 1.0)

Q = {json.loads(l)["qid"]: json.loads(l) for l in open(f"{SUB}/questions.jsonl", encoding="utf-8")}
P = {t: {json.loads(l)["qid"]: json.loads(l)["pred"]
         for l in open(f"{SUB}/preds_{t}.jsonl", encoding="utf-8")} for t in ("f16", "q4km")}

# ---- ① McNemar 精确检验(以"是否得满分"二值化) ----
b = c = 0   # b: f16 对 q4 错, c: f16 错 q4 对
for qid in Q:
    if qid not in P["f16"] or qid not in P["q4km"]: continue
    a1 = vqa_acc(P["f16"][qid], Q[qid]["answers"]) >= 0.999
    a2 = vqa_acc(P["q4km"][qid], Q[qid]["answers"]) >= 0.999
    if a1 and not a2: b += 1
    elif a2 and not a1: c += 1
n = b + c
k = max(b, c)
p_two = min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / (2 ** n)) if n else 1.0
print(f"① McNemar 精确检验:f16独对={b} q4km独对={c} 分歧={n}  双侧 p={p_two:.3f}")
print(f"   → {'差异不显著(无法区分两者)' if p_two > 0.05 else '差异显著'}")

# ---- ② 截断污染面 ----
def looks_truncated(s):
    s = s.rstrip()
    if not s: return False
    # 16 token 上限下,句末无终止符且长度较长 → 疑似截断
    return len(s) > 25 and s[-1] not in ".!?\"'" and not s.endswith(("s", "e", "y", "n"))

for t in ("f16", "q4km"):
    lens = [len(v) for v in P[t].values()]
    trunc = [qid for qid, v in P[t].items() if looks_truncated(v)]
    zero_but_contains = 0   # 得 0 分但预测里"包含"某个参考答案 → 典型截断/啰嗦受害者
    for qid, v in P[t].items():
        if vqa_acc(v, Q[qid]["answers"]) < 0.001:
            nv = norm(v)
            if any(norm(a) and norm(a) in nv for a in Q[qid]["answers"]):
                zero_but_contains += 1
    print(f"② {t:5s} 平均答案长度={sum(lens)/len(lens):.1f} 字符  疑似截断={len(trunc)}  "
          f"得0分但含正确答案={zero_but_contains}")

# ---- ③ 宽松口径(包含即算对)重评,看排序是否翻转 ----
for t in ("f16", "q4km"):
    loose = []
    for qid, v in P[t].items():
        nv = norm(v)
        hit = any(norm(a) and norm(a) in nv for a in Q[qid]["answers"])
        loose.append(1.0 if hit else vqa_acc(v, Q[qid]["answers"]))
    print(f"③ {t:5s} 宽松口径(答案被包含即算对) acc={sum(loose)/len(loose)*100:.2f}%")
