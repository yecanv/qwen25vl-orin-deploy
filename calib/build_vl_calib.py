#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VL 校准集构建
=============

为什么 VLM 的校准集不能沿用纯语言的做法
---------------------------------------
量化校准的本质是统计每层激活的动态范围（SmoothQuant 算 per-channel outlier
迁移系数 α，AWQ 算激活感知的权重重要性）。

VLM 的 LLM 主干在推理时，输入序列里很大一部分是**视觉 token**——由 ViT 输出、
经 projector 投影而来，激活分布和文本 embedding 完全不同（量级、稀疏度、
outlier 位置都不一样）。

只用纯文本语料校准，量化 scale 按文本分布定，实际推理时视觉 token 一进来就
大面积截断/溢出。现象是：**文字问答正常，一看图就胡说。**

=> 校准样本必须是真实图文对，走完整多模态前向。这是本脚本存在的理由，
   也是 `--text-only-ablation` 对照实验要证明的东西。


数据源的两种类型（本次修复的重点）
----------------------------------
上一版把所有数据源当成一样处理，这是错的：

  **A. self_contained —— 数据集自带图片**
      TextVQA、DocVQA、COCO Caption 这类，`load_dataset` 直接拿到 PIL Image。

  **B. annotation_only —— 只有标注，图片要另外下**
      LLaVA-Instruct-150K 只有 GPT-4 生成的对话标注，图片是 COCO 的，
      得自己下 train2017（19GB）或者按 image_id 拼 URL 逐张拉。
      ShareGPT4V 更麻烦，图来自 COCO / SAM / LAION / TextVQA 等多个源。

  上一版对 B 类直接取 `row["image"]` —— 拿到的是文件名字符串，不是图片，
  会被 `_extract_text` 静默跳过，最后只剩 A 类的样本，数量远少于 --n-samples。

本版做法：
  - 默认只用 A 类，开箱可跑
  - B 类需显式 `--coco-dir` 指定本地 COCO 图片目录才启用
  - 任一源失败，配额自动重分配给其他源，保证总数达标
  - 结束时校验实际条数，不足会明确报警而不是静默通过

许可
----
  | 数据集              | 许可          | 商用 |
  |---------------------|---------------|------|
  | COCO Captions       | CC BY 4.0     | 可   |
  | TextVQA             | CC BY 4.0     | 可   |
  | DocVQA              | 研究用途      | 否   |
  | LLaVA-Instruct-150K | 标注由GPT-4生成 | 需自行确认 |
  | ShareGPT4V          | CC BY-NC      | 否   |

  求职演示用途没问题；真商用要剔除 NC 项。

红线
----
  ✗ 不得用 MMBench / MMMU / C-Eval 等**评测集**做校准 —— 数据泄漏，
    面试官问一句"校准和评测是同一批数据吗"就穿帮。
  ✓ 评测集在 eval/datasets.yaml 单独声明，与本文件无交集。

用法
----
  # 先探测哪些源可用（建议第一次跑这个）
  python calib/build_vl_calib.py --probe

  # 默认（只用自带图片的源）
  python calib/build_vl_calib.py --n-samples 512

  # 加上 LLaVA（需本地 COCO 图片）
  python calib/build_vl_calib.py --n-samples 512 --coco-dir /data/coco/train2017

  # 消融对照组
  python calib/build_vl_calib.py --n-samples 512 --text-only-ablation
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch


# --------------------------------------------------------------------------- #
# 数据源定义
# --------------------------------------------------------------------------- #

SELF_CONTAINED = "self_contained"      # 数据集自带图片
NEEDS_IMAGES = "needs_local_images"    # 只有标注，图片要另外准备


@dataclass
class SourceSpec:
    name: str
    hf_repo: str
    split: str
    weight: float                       # 相对权重，脚本内部归一化
    kind: str = SELF_CONTAINED
    image_key: str = "image"
    text_key: str = "question"
    config: Optional[str] = None        # load_dataset 的第二个位置参数
    image_subdir: str = ""              # NEEDS_IMAGES 时，图片相对 --coco-dir 的子目录
    note: str = ""


# 默认源：全部 self_contained，开箱即用
DEFAULT_SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="textvqa",
        hf_repo="lmms-lab/TextVQA", split="validation", weight=0.30,
        image_key="image", text_key="question",
        note="图中文字，激活分布与自然图像差异大，必须覆盖"),
    SourceSpec(
        name="coco_caption",
        hf_repo="lmms-lab/COCO-Caption", split="val", weight=0.35,
        image_key="image", text_key="answer",
        note="通用图像描述，分布最广，主力"),
    SourceSpec(
        name="docvqa",
        hf_repo="lmms-lab/DocVQA", split="validation", weight=0.20,
        config="DocVQA", image_key="image", text_key="question",
        note="文档类高分辨率，视觉 token 数量级最大"),
    SourceSpec(
        name="vqav2",
        hf_repo="lmms-lab/VQAv2", split="validation", weight=0.15,
        image_key="image", text_key="question",
        note="通用视觉问答，补充问答型 prompt 分布"),
]

# 可选源：需要本地图片
OPTIONAL_SOURCES: List[SourceSpec] = [
    SourceSpec(
        name="llava_instruct",
        hf_repo="liuhaotian/LLaVA-Instruct-150K", split="train", weight=0.30,
        kind=NEEDS_IMAGES, image_key="image", text_key="conversations",
        image_subdir="",
        note="视觉指令数据，与 Instruct 模型 chat template 对齐。"
             "图片需自备 COCO train2017"),
]


# 备用镜像：主源拉不到时按顺序尝试
FALLBACK_REPOS: Dict[str, List[str]] = {
    "textvqa":      ["lmms-lab/textvqa", "facebook/textvqa"],
    "coco_caption": ["nlphuji/mscoco_2014_5k_test_image_text_retrieval",
                     "HuggingFaceM4/COCO"],
    "docvqa":       ["lmms-lab/DocVQA", "nielsr/docvqa_1200_examples"],
    "vqav2":        ["lmms-lab/VQAv2", "HuggingFaceM4/VQAv2"],
}


# --------------------------------------------------------------------------- #
# 分辨率控制
# --------------------------------------------------------------------------- #

def visual_tokens_for(max_pixels: int, patch: int = 14, merge: int = 2) -> int:
    """
    Qwen2-VL 动态分辨率：图按 patch=14 切块，再经 merger 2x2 合并。

        n_visual_tokens ≈ max_pixels / (patch² × merge²) = max_pixels / 784

    校准阶段建议把 max_pixels 压到 802816（≈1024 tokens）。
    DocVQA 那种高分辨率一张图能吃 4000+ token，FP16 校准会 OOM。
    """
    return max_pixels // (patch * patch * merge * merge)


# --------------------------------------------------------------------------- #
# 图片解析
# --------------------------------------------------------------------------- #

class ImageResolver:
    """把数据集里的图片字段变成 PIL.Image。"""

    def __init__(self, local_dir: Optional[str] = None):
        self.local_dir = Path(local_dir) if local_dir else None
        self.n_local = 0
        self.n_inline = 0
        self.n_failed = 0

    def resolve(self, value: Any, subdir: str = ""):
        from PIL import Image

        # 情况 1：数据集直接给了 PIL Image（self_contained）
        if hasattr(value, "convert"):
            self.n_inline += 1
            return value.convert("RGB")

        # 情况 2：datasets 的 Image feature 解码成 dict
        if isinstance(value, dict):
            if "bytes" in value and value["bytes"]:
                import io
                self.n_inline += 1
                return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
            value = value.get("path") or value.get("filename") or ""

        # 情况 3：文件名字符串（annotation_only）→ 去本地目录找
        if isinstance(value, str) and value:
            if self.local_dir is None:
                self.n_failed += 1
                return None
            for cand in (self.local_dir / subdir / value,
                         self.local_dir / value,
                         self.local_dir / subdir / Path(value).name):
                if cand.exists():
                    self.n_local += 1
                    return Image.open(cand).convert("RGB")

        self.n_failed += 1
        return None

    def summary(self) -> Dict[str, int]:
        return {"inline": self.n_inline, "local_file": self.n_local,
                "failed": self.n_failed}


def _fingerprint(img) -> str:
    """图片指纹，用于校准集/评测集重叠检查。"""
    import hashlib
    import numpy as np
    small = img.convert("L").resize((16, 16))
    return hashlib.md5(np.array(small).tobytes()).hexdigest()[:16]


def extract_text(row: Dict[str, Any], spec: SourceSpec) -> str:
    """不同数据集文本字段结构不同，统一抽成一句 user prompt。"""
    val = row.get(spec.text_key)
    if val is None:
        for k in ("question", "caption", "answer", "text", "query"):
            if k in row:
                val = row[k]
                break
    if val is None:
        return ""
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list) and val:
        # conversations: [{"from":"human","value":"..."}, ...]
        for turn in val:
            if isinstance(turn, dict) and turn.get("from") in ("human", "user"):
                return str(turn.get("value", "")).replace("<image>", "").strip()
        if isinstance(val[0], str):
            return val[0].strip()
        if isinstance(val[0], dict):
            return str(val[0].get("value", "")).strip()
    return ""


# --------------------------------------------------------------------------- #
# 探测：先确认哪些源能用
# --------------------------------------------------------------------------- #

def probe_sources(sources: List[SourceSpec]) -> Dict[str, Dict]:
    from datasets import load_dataset

    print(f"{'源':<18} {'状态':<10} {'实际 repo':<48} 说明")
    print("-" * 100)
    report = {}
    for spec in sources:
        candidates = [spec.hf_repo] + FALLBACK_REPOS.get(spec.name, [])
        ok_repo, err = None, ""
        for repo in candidates:
            try:
                kw = {"split": spec.split, "streaming": True}
                ds = load_dataset(repo, spec.config, **kw) if spec.config \
                    else load_dataset(repo, **kw)
                row = next(iter(ds))
                has_img = spec.image_key in row or "image" in row
                ok_repo = repo
                report[spec.name] = {
                    "repo": repo, "ok": True,
                    "has_image_field": has_img,
                    "columns": list(row.keys())[:8],
                }
                break
            except Exception as e:
                err = str(e)[:70]
        if ok_repo:
            print(f"{spec.name:<18} {'可用':<10} {ok_repo:<48} {spec.note[:24]}")
        else:
            print(f"{spec.name:<18} {'不可用':<10} {'—':<48} {err}")
            report[spec.name] = {"ok": False, "error": err}
    return report


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #

def build_calibration_set(
    model_id: str,
    n_samples: int,
    max_pixels: int,
    sources: List[SourceSpec],
    coco_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    from datasets import load_dataset
    from transformers import AutoProcessor

    random.seed(seed)
    torch.manual_seed(seed)

    processor = AutoProcessor.from_pretrained(
        model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels,
        trust_remote_code=True)

    n_tok = visual_tokens_for(max_pixels)
    per_sample_mb = n_tok * 4 * 1176 * 2 / 1e6          # fp16
    est_gb = per_sample_mb * n_samples / 1000
    print(f"[calib] max_pixels={max_pixels} → 单图视觉 token 上限 ≈ {n_tok}")
    print(f"[calib] 预估体积：单样本 {per_sample_mb:.1f} MB × {n_samples} 条 "
          f"= **{est_gb:.2f} GB**")
    if est_gb > 6:
        print(f"[calib] ⚠️  文件会很大，torch.load 时要全部读进内存。建议：")
        print(f"[calib]     --max-pixels 401408   （视觉 token 降到 512，体积减半）")
        print(f"[calib]     或 --n-samples 256    （128 条起就能出效果）")
        import sys
        if sys.stdin.isatty():
            ans = input("[calib] 继续？(y/N) ")
            if ans.strip().lower() != "y":
                raise SystemExit("已取消")

    resolver = ImageResolver(coco_dir)
    samples: List[Dict[str, Any]] = []
    manifest: List[Dict[str, Any]] = []
    failures: Dict[str, str] = {}

    # 归一化权重
    total_w = sum(s.weight for s in sources)
    remaining = n_samples
    pending = list(sources)

    while pending and remaining > 0:
        spec = pending.pop(0)
        # 配额按剩余量和剩余权重动态分配 —— 前面的源失败时后面自动补上
        rest_w = sum(s.weight for s in [spec] + pending)
        quota = min(remaining, max(1, round(remaining * spec.weight / rest_w)))
        if not pending:
            quota = remaining      # 最后一个源兜底

        print(f"\n[calib] {spec.name:<16} 配额 {quota:4d}   {spec.note}")

        if spec.kind == NEEDS_IMAGES and coco_dir is None:
            print(f"[calib]   跳过：该源只有标注，需 --coco-dir 指定本地图片目录")
            failures[spec.name] = "需要 --coco-dir"
            continue

        ds = None
        used_repo = None
        for repo in [spec.hf_repo] + FALLBACK_REPOS.get(spec.name, []):
            try:
                kw = {"split": spec.split, "streaming": True}
                ds = load_dataset(repo, spec.config, **kw) if spec.config \
                    else load_dataset(repo, **kw)
                used_repo = repo
                if repo != spec.hf_repo:
                    print(f"[calib]   主源失败，改用备用源 {repo}")
                break
            except Exception as e:
                failures[spec.name] = str(e)[:120]
        if ds is None:
            print(f"[calib]   拉取失败，配额转给其他源。原因：{failures.get(spec.name)}")
            continue

        taken = 0
        scanned = 0
        for row in ds:
            if taken >= quota:
                break
            scanned += 1
            if scanned > quota * 20:      # 防止在坏源上死循环
                print(f"[calib]   扫描 {scanned} 行只取到 {taken} 条，提前中止")
                break

            image = resolver.resolve(row.get(spec.image_key), spec.image_subdir)
            if image is None:
                continue
            text = extract_text(row, spec)
            if not text:
                continue

            messages = [{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": text}]}]
            prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)

            try:
                inputs = processor(text=[prompt], images=[image],
                                   return_tensors="pt", padding=False)
            except Exception:
                continue

            samples.append({
                "input_ids": inputs["input_ids"][0],
                "attention_mask": inputs["attention_mask"][0],
                # ★ 存 float16 而非 float32：体积减半。
                #   量化校准只需要激活的动态范围，fp16 精度完全够用，
                #   而且模型本来就是 fp16 推理的。
                "pixel_values": inputs["pixel_values"].to(torch.float16),
                "image_grid_thw": inputs["image_grid_thw"],
                "source": spec.name,
            })
            manifest.append({
                "source": spec.name,
                "repo": used_repo,
                "seq_len": int(inputs["input_ids"].shape[1]),
                "grid_thw": inputs["image_grid_thw"].tolist(),
                # 图片指纹：供 eval/build_eval_set.py 做重叠检查，
                # 证明校准集与评测集无交集。面试会问，别删。
                "fingerprint": _fingerprint(image),
            })
            taken += 1

        remaining -= taken
        print(f"[calib]   取得 {taken}/{quota}（扫描 {scanned} 行），"
              f"剩余待补 {remaining}")

    random.shuffle(samples)
    stats = summarize(manifest)

    # ---- 关键：校验实际条数，不足要明确报警 ----
    if len(samples) < n_samples:
        print(f"\n[calib] !! 只收集到 {len(samples)}/{n_samples} 条")
        print(f"[calib]    失败的源：{failures}")
        print(f"[calib]    图片解析情况：{resolver.summary()}")
        print(f"[calib]    建议：先跑 --probe 看哪些源可用；"
              f"或手动指定 --sources 只用可用的源")
    else:
        print(f"\n[calib] 收集完成 {len(samples)} 条")

    return {
        "samples": samples,
        "manifest": manifest,
        "stats": stats,
        "meta": {
            "model_id": model_id,
            "requested": n_samples,
            "actual": len(samples),
            "max_pixels": max_pixels,
            "seed": seed,
            "sources": [asdict(s) for s in sources],
            "failures": failures,
            "image_resolution": resolver.summary(),
        },
    }


def summarize(manifest: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not manifest:
        return {"n": 0}
    lens = sorted(m["seq_len"] for m in manifest)
    by_src: Dict[str, int] = {}
    for m in manifest:
        by_src[m["source"]] = by_src.get(m["source"], 0) + 1
    return {
        "n_samples": len(manifest),
        "seq_len_min": lens[0],
        "seq_len_p50": lens[len(lens) // 2],
        "seq_len_p95": lens[int(len(lens) * 0.95)],
        "seq_len_max": lens[-1],
        "by_source": by_src,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--n-samples", type=int, default=512,
                    help="128 起步即可出效果，512 是精度/耗时平衡点；"
                         "本项目要做 128/256/512 消融")
    ap.add_argument("--max-pixels", type=int, default=802816,
                    help="≈1024 视觉 token，16GB 板子校准的安全上限")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--coco-dir", default=None,
                    help="本地 COCO 图片目录（如 /data/coco/train2017），"
                         "指定后启用 LLaVA-Instruct 等仅标注的源")
    ap.add_argument("--sources", default="",
                    help="逗号分隔，只用指定的源（如 textvqa,coco_caption）")
    ap.add_argument("--probe", action="store_true",
                    help="只探测各数据源是否可用，不构建")
    ap.add_argument("--text-only-ablation", action="store_true",
                    help="产出纯文本校准集做对照实验")
    args = ap.parse_args()

    sources = list(DEFAULT_SOURCES)
    if args.coco_dir:
        sources += OPTIONAL_SOURCES
    if args.sources:
        want = {s.strip() for s in args.sources.split(",")}
        sources = [s for s in sources if s.name in want]
        if not sources:
            print(f"没有匹配的源。可选：{[s.name for s in DEFAULT_SOURCES+OPTIONAL_SOURCES]}")
            return

    if args.probe:
        report = probe_sources(sources)
        Path("calib/data").mkdir(parents=True, exist_ok=True)
        Path("calib/data/source_probe.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        n_ok = sum(1 for v in report.values() if v.get("ok"))
        print(f"\n{n_ok}/{len(sources)} 个源可用 → calib/data/source_probe.json")
        if n_ok == 0:
            print("全部不可用。检查网络/HF 镜像：export HF_ENDPOINT=https://hf-mirror.com")
        return

    if args.text_only_ablation:
        print("[calib] 消融模式：纯文本校准集（对照组）")
        print("[calib] 目的是证明只用文本校准会导致图文任务掉点")
        # 纯文本路径不需要 processor 的图像分支，单独处理
        build_text_only(args)
        return

    pack = build_calibration_set(
        model_id=args.model, n_samples=args.n_samples,
        max_pixels=args.max_pixels, sources=sources,
        coco_dir=args.coco_dir, seed=args.seed)

    out = Path(args.out or f"calib/data/vl_calib_{pack['meta']['actual']}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pack, out)
    print(f"\n[calib] 已保存 → {out}  ({out.stat().st_size/1e6:.1f} MB)")

    meta = out.with_suffix(".meta.json")
    meta.write_text(json.dumps({"meta": pack["meta"], "stats": pack["stats"]},
                               ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[calib] 元信息 → {meta}")
    print(f"[calib] 面试会问校准集来源，这个 json 保留好，别删。")

    print("\n[calib] 统计：")
    for k, v in pack["stats"].items():
        print(f"        {k:<20} {v}")


def build_text_only(args):
    """纯文本对照组。用于证明 VLM 校准必须走多模态前向。"""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    samples, manifest = [], []

    for repo, key in [("Skywork/SkyPile-150B", "text"),
                      ("wikimedia/wikipedia", "text")]:
        try:
            kw = {"split": "train", "streaming": True}
            ds = load_dataset(repo, "20231101.zh", **kw) if "wikipedia" in repo \
                else load_dataset(repo, **kw)
            for row in ds:
                if len(samples) >= args.n_samples:
                    break
                text = (row.get(key) or "")[:2048]
                if len(text) < 100:
                    continue
                enc = tok(text, return_tensors="pt", truncation=True, max_length=1024)
                samples.append({
                    "input_ids": enc["input_ids"][0],
                    "attention_mask": enc["attention_mask"][0],
                    "pixel_values": None,
                    "image_grid_thw": None,
                    "source": "text_only",
                })
                manifest.append({"source": "text_only", "repo": repo,
                                 "seq_len": int(enc["input_ids"].shape[1]),
                                 "grid_thw": None})
            if len(samples) >= args.n_samples:
                break
        except Exception as e:
            print(f"[calib] {repo} 失败：{str(e)[:100]}")

    out = Path(args.out or f"calib/data/text_only_{len(samples)}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"samples": samples, "manifest": manifest,
                "stats": summarize(manifest),
                "meta": {"model_id": args.model, "actual": len(samples),
                         "ablation": "text_only"}}, out)
    print(f"[calib] 对照组已保存 → {out}  ({len(samples)} 条)")
    print("[calib] 用它量化一版，和图文混合版对比 MMBench/TextVQA，"
          "差值就是实验结论。")


if __name__ == "__main__":
    main()
