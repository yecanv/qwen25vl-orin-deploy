#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果汇总：出图 + 自动填数据表
=============================

读 results/raw/ 下所有 json，产出：
  1. results/figures/*.png   —— 论文级图表
  2. results/RESULTS_FILLED.md —— 把模板里的空格填上

设计原则：**没跑的测试不填，留空并标注**。
自动编数据比留空危险得多——你会不知道哪个数字是真的。
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, Any, Optional

RAW = Path("results/raw")
FIG = Path("results/figures")


def load(name: str) -> Optional[Dict[str, Any]]:
    p = RAW / name
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[warn] {p} 解析失败: {e}")
        return None


def find_all(pattern: str):
    return sorted(RAW.glob(pattern))


# --------------------------------------------------------------------------- #
# 出图
# --------------------------------------------------------------------------- #

def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # 注意：rcParams 赋值本身不校验字体是否存在（不会抛异常），
    # 必须用 font_manager 查已安装字体再选。候选覆盖
    # Windows（微软雅黑/黑体）与 Jetson/Linux（Noto/文泉驿）。
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for f in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
              "Noto Sans CJK JP", "WenQuanYi Zen Hei"]:
        if f in installed:
            matplotlib.rcParams["font.sans-serif"] = [f, "DejaVu Sans"]
            break
    else:
        print("[warn] 未找到中文字体，图中中文会显示为方框")
    matplotlib.rcParams["axes.unicode_minus"] = False
    return plt


def plot_ttft_breakdown(plt):
    """TTFT 分解：视觉编码 vs LLM prefill。项目最核心的一张图。"""
    files = find_all("latency_*.json")
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        tag = d.get("tag", f.stem)
        rs = d.get("results", [])
        if not rs:
            continue
        x = [r["n_visual_tokens"] for r in rs]
        vit = [r["vit_ms"]["p50"] for r in rs]
        llm = [r["llm_prefill_ms"]["p50"] for r in rs]
        ax.plot(x, vit, "o-", label=f"{tag} ViT编码")
        ax.plot(x, llm, "s--", label=f"{tag} LLM prefill")
    ax.set_xlabel("视觉 token 数")
    ax.set_ylabel("延迟 (ms)")
    ax.set_title("TTFT 分解：视觉编码 vs 语言 prefill")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    p = FIG / "ttft_breakdown.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_quant_comparison(plt):
    """三种量化方案的显存-延迟-精度对比。"""
    files = find_all("latency_*.json")
    if len(files) < 2:
        return None
    tags, ttfts, decodes = [], [], []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rs = d.get("results", [])
        if not rs:
            continue
        mid = rs[len(rs) // 2]
        tags.append(d.get("tag", f.stem))
        ttfts.append(mid["e2e_ttft_ms"]["p50"])
        decodes.append(mid["decode_tok_s"]["p50"])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    a1.bar(tags, ttfts, color="#4C72B0"); a1.set_ylabel("端到端 TTFT (ms)")
    a1.set_title("量化方案 vs TTFT"); a1.grid(axis="y", alpha=0.3)
    a2.bar(tags, decodes, color="#DD8452"); a2.set_ylabel("decode (tok/s)")
    a2.set_title("量化方案 vs 解码吞吐"); a2.grid(axis="y", alpha=0.3)
    p = FIG / "quant_comparison.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_sensitivity(plt):
    """
    量化敏感度散点图：X = 参数占比（收益），Y = KL 散度（代价）。
    右下角该量化，左上角不该量化。这是"为什么 ViT 不量化"的证据图。
    """
    d = load("sensitivity.json")
    if not d:
        return None
    groups = d.get("groups", [])
    if not groups:
        return None

    fig, ax = plt.subplots(figsize=(8, 5.5))
    KL_FLOOR = 1e-6   # KL 为负/近零是采样噪声，对数轴画不了，钳到地板值并标注
    for g in groups:
        is_vision = "vision" in g["group"]
        kl = g.get("kl_int8")
        if kl is None:
            continue
        clamped = kl < KL_FLOOR
        kl_plot = max(kl, KL_FLOOR)
        ax.scatter(g["param_share_pct"], kl_plot,
                   s=90, marker="^" if is_vision else "o",
                   color="#C44E52" if is_vision else "#55A868",
                   edgecolors="k", linewidths=0.6, zorder=3)
        label = g["group"] + ("（KL≈0）" if clamped else "")
        ax.annotate(label, (g["param_share_pct"], kl_plot),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("参数量占比 (%)  →  量化收益")
    ax.set_ylabel("INT8 伪量化 KL 散度  →  量化代价")
    ax.set_yscale("log")
    ax.set_title("逐模块量化敏感度\n（▲ 视觉塔  ● LLM 主干）")
    ax.grid(alpha=0.3, zorder=0)
    p = FIG / "quant_sensitivity.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_stability(plt):
    """长稳：延迟漂移与结温。车端最有说服力的一张图。"""
    d = load("stability.json")
    if not d or not d.get("windows"):
        return None
    w = d["windows"]
    x = list(range(len(w)))
    labels = [ww["window"] for ww in w]

    fig, ax1 = plt.subplots(figsize=(9, 4.5))
    ax1.plot(x, [ww["p50_ms"] for ww in w], "o-", label="P50", color="#4C72B0")
    ax1.plot(x, [ww["p95_ms"] for ww in w], "s--", label="P95", color="#DD8452")
    ax1.plot(x, [ww["p99_ms"] for ww in w], "^:", label="P99", color="#C44E52")
    ax1.set_ylabel("延迟 (ms)"); ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, fontsize=8)
    ax1.grid(alpha=0.3); ax1.legend(loc="upper left", fontsize=8)

    temps = [ww.get("tj_mean_c") for ww in w]
    if any(t is not None for t in temps):
        ax2 = ax1.twinx()
        ax2.plot(x, temps, "d-.", color="#8172B3", label="结温")
        ax2.set_ylabel("结温 (°C)")
        ax2.legend(loc="upper right", fontsize=8)

    drift = d.get("p50_drift_pct")
    ax1.set_title(f"30min 长稳测试（P50 漂移 {drift}%）")
    p = FIG / "stability.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_throughput(plt):
    files = find_all("throughput_*.json")
    if not files:
        return None
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        rs = d.get("results", [])
        if not rs:
            continue
        tag = d.get("tag", f.stem)
        c = [r["concurrency"] for r in rs]
        a1.plot(c, [r["throughput_tok_s"] for r in rs], "o-", label=tag)
        a2.plot(c, [r["ttft_p99_ms"] for r in rs], "s-", label=tag)
    a1.set_xlabel("并发数"); a1.set_ylabel("吞吐 (tok/s)"); a1.grid(alpha=0.3)
    a1.set_title("并发 vs 吞吐"); a1.legend(fontsize=8)
    a2.set_xlabel("并发数"); a2.set_ylabel("TTFT P99 (ms)"); a2.grid(alpha=0.3)
    a2.set_title("并发 vs 尾延迟"); a2.legend(fontsize=8)
    p = FIG / "throughput.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_kernel(plt):
    d = load("kernel_bench.json")
    if not d or not d.get("kernel_level"):
        return None
    ks = d["kernel_level"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
    x = [k["n_tokens"] for k in ks]
    a1.plot(x, [k["naive_ms"] for k in ks], "o-", label="朴素三段式")
    a1.plot(x, [k["fused_ms"] for k in ks], "s-", label="融合算子")
    a1.set_xlabel("视觉 token 数"); a1.set_ylabel("kernel 耗时 (ms)")
    a1.set_title("融合 vs 朴素"); a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.plot(x, [k["speedup"] for k in ks], "^-", color="#55A868")
    a2.axhline(1.0, ls="--", c="gray")
    a2.set_xlabel("视觉 token 数"); a2.set_ylabel("加速比")
    a2.set_title("加速比随规模变化\n(N小→launch开销主导, N大→S矩阵访存主导)")
    a2.grid(alpha=0.3)
    p = FIG / "kernel_fusion.png"
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# 填表
# --------------------------------------------------------------------------- #

def fill_template() -> Path:
    tpl = Path("results/RESULTS_TEMPLATE.md")
    if not tpl.exists():
        print("[warn] 找不到模板")
        return None
    text = tpl.read_text(encoding="utf-8")

    dev = None
    p = Path("results/device_info.json")
    if p.exists():
        dev = json.loads(p.read_text(encoding="utf-8"))

    filled, missing = 0, []

    # 环境表
    if dev:
        rows = {
            "板卡型号": dev.get("model", ""),
            "内存": f"{dev.get('mem_total_mb','?')} MB",
            "JetPack / L4T": f"{dev.get('jetpack','?')} / {dev.get('l4t','?')}",
            "CUDA / TensorRT / TRT-LLM":
                f"{dev.get('cuda','?')} / {dev.get('tensorrt','?')} / {dev.get('tensorrt_llm','?')}",
            "功耗模式（nvpmodel）": dev.get("nvpmodel", ""),
            "实测带宽 (GB/s)": str(dev.get("measured_bandwidth_gbs", "")),
        }
        for k, v in rows.items():
            if not v or v.strip() in ("?", "/ ?"):
                continue
            pat = re.compile(rf"^\| {re.escape(k)} \| *\|$", re.M)
            new, n = pat.subn(f"| {k} | {v} |", text)
            if n:
                text = new; filled += n
    else:
        missing.append("results/device_info.json（跑 benchmark/probe_device.py）")

    # 长稳
    st = load("stability.json")
    if st:
        text = text.replace("- P50 漂移：___%", f"- P50 漂移：{st.get('p50_drift_pct')}%")
        text = text.replace("- 结温峰值：___°C",
                            f"- 结温峰值：{st['thermal']['tj_max_c']}°C")
        text = text.replace("- 错误数：___", f"- 错误数：{st['n_errors']}")
        filled += 3
        w = st.get("windows", [])
        lines = []
        for ww in w:
            lines.append(f"| {ww['window']} | {ww['n']} | {ww['p50_ms']} | "
                         f"{ww['p95_ms']} | {ww['p99_ms']} | {ww.get('tj_mean_c','')} | |")
        if lines:
            text = re.sub(r"(\| 时间窗 \| 请求数 \| P50 \| P95 \| P99 \| 结温 \| GPU 频率 \|\n\|[-| ]+\|\n)(\|[^\n]*\|\n)+",
                          r"\1" + "\n".join(lines) + "\n", text)
            filled += len(lines)
    else:
        missing.append("results/raw/stability.json（跑 benchmark/bench_power_stability.py）")

    # TTFT 分解表
    for f in find_all("latency_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        lines = []
        for r in d.get("results", []):
            lines.append(f"| {r['resolution_label']} | {r['n_visual_tokens']} | "
                         f"{r['vit_ms']['p50']} | {r['vit_share_pct']}% | "
                         f"{r['llm_prefill_ms']['p50']} | {r['e2e_ttft_ms']['p50']} |")
        if lines:
            text = re.sub(r"(\| 分辨率 \| 视觉 token \| ViT \(ms\) \| 占 TTFT \| LLM prefill \(ms\) \| e2e TTFT \(ms\) \|\n\|[-| ]+\|\n)(\|[^\n]*\|\n)+",
                          r"\1" + "\n".join(lines) + "\n", text)
            filled += len(lines)
        break
    else:
        missing.append("results/raw/latency_*.json（跑 benchmark/bench_latency.py）")

    # 敏感度表
    sen = load("sensitivity.json")
    if sen:
        lines = []
        for g in sen.get("groups", []):
            mark = "✗" if "vision" in g["group"] or "lm_head" in g["group"] else "✓"
            lines.append(f"| {g['group']} | {g['param_share_pct']}% | "
                         f"{g['outlier_ratio_max']} | {g.get('kl_int8','—')} | {mark} |")
        if lines:
            text = re.sub(r"(\| 模块组 \| 参数占比 \| outlier ratio \(max\) \| INT8 伪量化 KL \| 是否量化 \|\n\|[-| ]+\|\n)(\|[^\n]*\|\n)+",
                          r"\1" + "\n".join(lines) + "\n", text)
            filled += len(lines)
    else:
        missing.append("results/raw/sensitivity.json（跑 quantize/sensitivity_analysis.py）")

    # 并发表
    for f in find_all("throughput_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        lines = []
        for r in d.get("results", []):
            note = f"{r['n_errors']} 失败" if r["n_errors"] else ""
            lines.append(f"| {r['concurrency']} | {r['throughput_tok_s']} | "
                         f"{r['ttft_p50_ms']} | {r['ttft_p99_ms']} | "
                         f"{r['mem_peak_mb']}MB | {note} |")
        if lines:
            text = re.sub(r"(\| 并发数 \| 吞吐 \(tok/s\) \| P50 TTFT \| P99 TTFT \| 内存峰值 \| 备注 \|\n\|[-| ]+\|\n)(\|[^\n]*\|\n)+",
                          r"\1" + "\n".join(lines) + "\n", text)
            filled += len(lines)
        break

    # 功耗
    if st:
        pw, ef = st["power"], st["efficiency"]
        text = text.replace("| FP16 | | | | | |",
            f"| （当前方案） | {pw['vdd_in_mean_mw']/1000:.2f} W | — | "
            f"{pw['vdd_in_peak_mw']/1000:.2f} W | {pw['energy_total_j']} J | "
            f"{ef['tokens_per_joule_vddin']} |", 1)
        filled += 1

    out = Path("results/RESULTS_FILLED.md")
    header = ("> 本文件由 benchmark/plot_results.py 自动生成。\n"
              "> **仍为 `___` 或空白的格子说明对应测试没跑**，不要往里编数字。\n\n")
    if missing:
        header += "> 缺失的数据源：\n"
        for m in missing:
            header += f"> - {m}\n"
        header += "\n"
    out.write_text(header + text, encoding="utf-8")
    print(f"\n填表：填入 {filled} 处 → {out}")
    if missing:
        print("尚缺：")
        for m in missing:
            print(f"  - {m}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fill-template", action="store_true")
    ap.add_argument("--sensitivity", action="store_true", help="只出敏感度图")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    FIG.mkdir(parents=True, exist_ok=True)

    if not args.no_plot:
        try:
            plt = setup_mpl()
        except ImportError:
            print("未安装 matplotlib，跳过出图：pip install matplotlib")
            plt = None

        if plt:
            makers = [plot_sensitivity] if args.sensitivity else [
                plot_ttft_breakdown, plot_quant_comparison, plot_sensitivity,
                plot_stability, plot_throughput, plot_kernel,
            ]
            print("出图：")
            for fn in makers:
                try:
                    p = fn(plt)
                    print(f"  {'✓ ' + str(p) if p else '– ' + fn.__name__ + '（数据缺失，跳过）'}")
                except Exception as e:
                    print(f"  ✗ {fn.__name__}: {e}")

    if args.fill_template or not args.sensitivity:
        fill_template()


if __name__ == "__main__":
    main()
