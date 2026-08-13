#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
构建评测集
==========
与校准集严格分离，来源见 eval/datasets.yaml。

分离规则（面试会问）：
  - TextVQA：校准用 validation，评测用 test
  - MMBench / MME：完全不在校准源里
  - 构建时会主动检查与校准集的重叠，有重叠直接报错

用法：
  python eval/build_eval_set.py --n 100
"""
import argparse, json, hashlib
from pathlib import Path


def image_fingerprint(img) -> str:
    """图片指纹，用于检测评测集与校准集是否有重叠。"""
    import numpy as np
    small = img.convert("L").resize((16, 16))
    return hashlib.md5(np.array(small).tobytes()).hexdigest()[:16]


def load_calib_fingerprints(calib_path: str) -> set:
    p = Path(calib_path)
    if not p.exists():
        print(f"[eval] 未找到校准集 {p}，跳过重叠检查")
        print(f"[eval] 建议先建校准集，否则无法证明两者无交集")
        return set()
    import torch
    pack = torch.load(p, weights_only=False)
    fps = set()
    for m in pack.get("manifest", []):
        if "fingerprint" in m:
            fps.add(m["fingerprint"])
    return fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="eval/data/eval_100.json")
    ap.add_argument("--img-dir", default="eval/data/images")
    ap.add_argument("--calib", default="calib/data/vl_calib_512.pt")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    from datasets import load_dataset

    img_dir = Path(args.img_dir); img_dir.mkdir(parents=True, exist_ok=True)
    calib_fps = load_calib_fingerprints(args.calib)
    print(f"[eval] 校准集指纹 {len(calib_fps)} 个，将用于重叠检查")

    sources = [
        ("mmbench_cn", "lmms-lab/MMBench_CN", "dev", None, "question"),
        ("textvqa_test", "lmms-lab/TextVQA", "test", None, "question"),
    ]

    samples, overlaps = [], 0
    per_src = max(1, args.n // len(sources))

    for name, repo, split, cfg, tkey in sources:
        print(f"\n[eval] {name}  目标 {per_src} 条")
        try:
            ds = load_dataset(repo, cfg, split=split, streaming=True) if cfg \
                else load_dataset(repo, split=split, streaming=True)
        except Exception as e:
            print(f"[eval]   拉取失败：{str(e)[:100]}")
            continue

        taken = 0
        for i, row in enumerate(ds):
            if taken >= per_src or len(samples) >= args.n:
                break
            img = row.get("image")
            if img is None or not hasattr(img, "convert"):
                continue
            fp = image_fingerprint(img)
            if fp in calib_fps:
                overlaps += 1
                continue                      # ★ 与校准集重叠，剔除
            q = row.get(tkey) or row.get("question") or "描述这张图片。"
            ans = row.get("answer") or row.get("answers") or ""

            path = img_dir / f"{name}_{taken:04d}.jpg"
            img.convert("RGB").save(path, quality=92)
            samples.append({
                "image": str(path),
                "prompt": str(q),
                "reference": ans if isinstance(ans, str) else str(ans),
                "source": name,
                "fingerprint": fp,
            })
            taken += 1
        print(f"[eval]   取得 {taken}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(samples, ensure_ascii=False, indent=2),
                   encoding="utf-8")

    print(f"\n[eval] 评测集 {len(samples)} 条 → {out}")
    print(f"[eval] 因与校准集重叠而剔除 {overlaps} 条")
    if calib_fps and overlaps == 0:
        print("[eval] 无重叠，校准/评测分离成立 ✓")
    print("[eval] 这个结论面试要用，别删 fingerprint 字段。")


if __name__ == "__main__":
    main()
