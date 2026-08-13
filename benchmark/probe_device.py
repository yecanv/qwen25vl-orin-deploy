#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
板卡探测 —— 跑任何 benchmark 之前先跑这个
=========================================
目的：把"你在什么硬件、什么配置下测的"固化成一份可追溯的记录。
面试官问"你这个 18ms 什么环境测的"，答案就在这个文件里。
"""
import json, os, re, subprocess, sys
from pathlib import Path


def sh(cmd, default=""):
    try:
        return subprocess.check_output(cmd, shell=True, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return default


def probe():
    info = {}
    # --- 板卡型号 ---
    info["model"] = sh("cat /proc/device-tree/model 2>/dev/null | tr -d '\\0'", "unknown")
    info["l4t"] = sh("cat /etc/nv_tegra_release 2>/dev/null | head -1")
    info["jetpack"] = sh("dpkg-query -W -f='${Version}' nvidia-jetpack 2>/dev/null")

    # --- 功耗模式（对性能影响极大，必须记录）---
    info["nvpmodel"] = sh("nvpmodel -q 2>/dev/null | tail -2 | tr '\\n' ' '")
    info["jetson_clocks"] = sh("jetson_clocks --show 2>/dev/null | head -5")

    # --- 软件栈 ---
    info["cuda"] = sh("nvcc --version 2>/dev/null | grep release")
    try:
        import tensorrt
        info["tensorrt"] = tensorrt.__version__
    except Exception:
        info["tensorrt"] = sh("dpkg -l | grep -m1 libnvinfer | awk '{print $3}'")
    try:
        import tensorrt_llm
        info["tensorrt_llm"] = tensorrt_llm.__version__
    except Exception:
        info["tensorrt_llm"] = "not installed"
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["sm"] = f"{p.major}.{p.minor}"
            info["total_mem_mb"] = p.total_memory // 1024 // 1024
            info["sm_count"] = p.multi_processor_count
    except Exception as e:
        info["torch"] = f"error: {e}"

    # --- 内存 / swap ---
    info["mem_total_mb"] = int(sh("free -m | awk '/Mem:/{print $2}'", "0") or 0)
    info["swap_total_mb"] = int(sh("free -m | awk '/Swap:/{print $2}'", "0") or 0)

    # --- 实测带宽（决定 memory-bound 分析的基准线）---
    info["measured_bandwidth_gbs"] = measure_bandwidth()

    return info


def measure_bandwidth():
    """
    实测显存带宽。Orin 是统一内存，官方标称 102 GB/s（Nano/NX）或
    204.8 GB/s（AGX），但实际可达带宽通常只有标称的 70-85%。

    做 roofline 分析时必须用**实测值**，用标称值算出来的"带宽利用率"
    是假的，面试官一问就露。
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        n = 64 * 1024 * 1024              # 64M float16 = 128MB
        a = torch.empty(n, dtype=torch.float16, device="cuda")
        b = torch.empty(n, dtype=torch.float16, device="cuda")
        for _ in range(5):
            b.copy_(a)
        torch.cuda.synchronize()
        import time
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            b.copy_(a)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        bytes_moved = n * 2 * 2 * iters   # 读+写
        return round(bytes_moved / dt / 1e9, 1)
    except Exception:
        return None


if __name__ == "__main__":
    info = probe()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    out = Path("results/device_info.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ {out}")

    print("\n检查清单：")
    if "Orin" not in str(info.get("model", "")):
        print("  [!] 未检测到 Orin —— 你是在桌面机上跑吗？engine 不通用。")
    if info.get("swap_total_mb", 0) < 8000 and info.get("mem_total_mb", 0) < 10000:
        print("  [!] 8GB 板子 swap 不足 8G，build engine 可能 OOM，"
              "先跑 env/jetson_setup.sh")
    if "MAXN" not in str(info.get("nvpmodel", "")):
        print("  [!] 未处于 MAXN 模式，性能数据会偏低。"
              "记录下当前模式，benchmark 报告里要写。")
