# Qwen2.5-VL 边缘端多模态推理部署与量化压缩（Jetson Orin）

把多模态大模型压进车规级算力平台，并建立可追溯的性能评测体系。

---

## 1. 项目定位

| 维度 | 说明 |
|---|---|
| 目标 | 在 Jetson Orin 上跑通 VLM 端到端推理，视觉编码 FP16 + LLM 主干量化 |
| 模型 | Qwen2.5-VL-3B-Instruct（16GB 板）/ Qwen2-VL-2B-Instruct（8GB 板） |
| 引擎 | TensorRT（ViT）+ llama.cpp（LLM，Q4_K_M，混合链路已板上实测交付）；TensorRT-LLM 0.12 因 M-RoPE 兼容边界栈内端到端不通，留作外置 RoPE 补 0.12 的后续路线 |
| 量化 | INT8-SmoothQuant / INT4-AWQ，仅作用于 LLM 主干 |
| 评测 | TTFT / TPOT / 吞吐 / 显存峰值 / 冷启动 / 功耗 / 长稳抖动 / 精度回归 |

> **模型版本说明**：Qwen2.5-VL 官方只有 3B / 7B / 72B，**没有 2B**。
> 8GB 板子跑不动 3B 的话，退到 **Qwen2-VL-2B-Instruct**（架构同源，M-RoPE、
> patch merger 机制一致，本仓代码两条路径都支持，用 `--model-family` 切换）。

---

## 2. 为什么 VLM 部署比纯语言难

这三点是本项目的核心工程量：

**① 双引擎异构**
ViT 是定长/动态分辨率的视觉编码器，LLM 是自回归解码器，两者计算特征完全不同。
ViT 走标准 TensorRT engine，LLM 走 TensorRT-LLM，中间靠 **prompt table** 桥接。

**② M-RoPE（多模态旋转位置编码）**
Qwen2-VL / 2.5-VL 用的不是标准 1D RoPE，而是拆成 (temporal, height, width) 三段的
M-RoPE。图像 token 的 position_id 需要按 patch 网格二维展开，**算错了不会报错，
只会输出乱码**——这是最难 debug 的一类问题。

**③ 视觉 token 膨胀**
一张 1280×720 的图，先对齐到 28 像素倍数（1288×728），经 patch(14×14) + merger(2×2)
后产生 (1288/28)×(728/28) = 46×26 ≈ **1200 个视觉 token**，远超文本部分。这直接决定了 TTFT 和 KV Cache 显存占用，也是本项目"视觉 token 压缩"
优化的切入点。

---

## 3. 数据流

```
                  ┌─────────────────────────────────────────┐
   image ───────► │ ViT (FP16, TensorRT engine)             │
                  │  patch embed → window attn → merger     │
                  └──────────────┬──────────────────────────┘
                                 │ vision embeds [N_img, hidden]
                                 ▼
                  ┌─────────────────────────────────────────┐
   text  ───────► │ prompt table 拼接                        │
   tokens         │  fake_ids = vocab_size + [0..N_img)      │
                  │  M-RoPE position_ids 二维展开            │
                  └──────────────┬──────────────────────────┘
                                 ▼
                  ┌─────────────────────────────────────────┐
                  │ LLM 主干 (INT8-SQ / INT4-AWQ)            │
                  │  TensorRT-LLM + Paged KV Cache          │
                  │  inflight batching                      │
                  └──────────────┬──────────────────────────┘
                                 ▼
                            streaming output
```

---

## 4. 硬件适配矩阵

| 板卡 | 内存 | 带宽 | INT8 算力 | 建议模型 | 备注 |
|---|---|---|---|---|---|
| Orin Nano Super 8GB | 8GB LPDDR5 | 102 GB/s | ~67 TOPS | Qwen2-VL-2B (INT4) | 紧，需关桌面、开 swap |
| Orin NX 16GB Super | 16GB LPDDR5 | 102 GB/s | ~157 TOPS | Qwen2.5-VL-3B (INT8/INT4) | **推荐** |
| AGX Orin 32/64GB | 32/64GB | 204.8 GB/s | ~275 TOPS | Qwen2.5-VL-7B | 数据最好看，贵 |

**注意 Orin 是统一内存架构（UMA）**，没有独立显存，模型权重和系统内存抢同一块。
`tegrastats` 里的 RAM 就是全部，这点跟桌面卡完全不同，Benchmark 口径要写清楚。

> 表内规格以 NVIDIA 官方 datasheet 为准，**请在自己板子上用
> `benchmark/probe_device.py` 实测确认**，不同 JetPack 版本和功耗模式差异很大。

---

## 5. 快速开始

```bash
# 0. 准备：生成测试图 + 探测板卡
python assets/make_demo.py            # 程序化生成，可复现，无版权问题
python benchmark/probe_device.py      # 记录环境基线

# 1. 环境
sudo bash env/jetson_setup.sh          # 功耗模式 / swap / jetson_clocks
bash env/build_trtllm.sh               # TensorRT-LLM ARM64 编译（耗时 2-4h）

# 2. 校准集（公开数据，自动下载）
python calib/build_vl_calib.py --model Qwen/Qwen2.5-VL-3B-Instruct \
    --n-samples 512 --out calib/data/vl_calib_512.pt

# 3. ViT → ONNX → TRT engine (FP16)
python convert/export_vit_onnx.py --model Qwen/Qwen2.5-VL-3B-Instruct
bash convert/build_vit_engine.sh

# 4. LLM 主干量化 + engine
python convert/quantize_llm.py --qformat int4_awq --calib calib/data/vl_calib_512.pt
bash convert/build_llm_engine.sh

# 5. 跑起来
python runtime/run_vl.py --image assets/demo.jpg --prompt "描述这张图片"

# 6. 全套 benchmark（结果写入 results/raw/）
bash benchmark/run_all.sh
python benchmark/plot_results.py
```

---

## 6. 目录

```
├── docs/          环境搭建、模型转换、量化、Benchmark 方法论与事故报告
├── env/           Jetson 环境配置与 TRT-LLM 编译脚本
├── calib/         VL 校准集构建（公开数据集）
├── convert/       ViT ONNX 导出、LLM 量化、engine 构建
├── quantize/      逐层敏感度分析、量化方案对比
├── runtime/       推理编排（Python 参考实现：M-RoPE + 双引擎桥接；hybrid_driver.cpp：手写 C++ 混合链路驱动，已板上实测）
├── benchmark/     六类指标测量 + 出图
├── eval/          精度回归（VQA / 一致性 / PPL）
└── results/       原始日志、汇总表、图表（实测数据入账处）
```

---

## 7. 关于数据的说明

**代码、文档、校准集、脚本 —— 全部随仓交付。**

**Benchmark 数字均为本机实测。** 原因是技术性的：

1. TensorRT engine 是**硬件绑定**的。engine 里烧进了目标 SM 架构、显存容量、
   TensorRT 版本、cuDNN 版本。我机器上 build 的 engine，你板子上 `deserializeCudaEngine`
   直接返回 nullptr。
2. 即便同为 Orin，**功耗模式（15W/25W/MAXN）、JetPack 版本、是否 jetson_clocks、
   环境温度**都会让延迟差出 30% 以上。
3. 每个性能数字都应能回答“多少 ms、什么配置下测的”——本仓库用 results/DATA_LEDGER.md 保证每个数字可溯源。

`results/` 含数据表模板与指标定义；已实测入账的数据在 `results/verified/`（DATA_LEDGER 可溯源），未跑的测试留空。

---

## 8. 参考实现

- TensorRT-LLM multimodal examples: `examples/multimodal/`
- NVIDIA ModelOpt (量化后端): `nvidia-modelopt`
- jetson-containers (dusty-nv): Jetson 上的 TRT-LLM 预编译镜像
- Qwen2-VL / Qwen2.5-VL 技术报告（M-RoPE、动态分辨率、window attention 章节）
