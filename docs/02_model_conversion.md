# 模型转换与 Engine 构建

## 整体流程

```
Qwen2.5-VL-3B (HuggingFace)
        │
        ├─── 视觉塔 ────► ONNX ────► TensorRT engine (FP16)     [在 Orin 上 build]
        │                  ▲
        │            export_vit_onnx.py
        │
        └─── LLM 主干 ──► 量化 ────► TRT-LLM ckpt ──► engine     [量化在桌面卡]
                            ▲                          ▲          [build 在 Orin]
                     quantize_llm.py            trtllm-build
```

**为什么量化和 build 分在两台机器上**

| 步骤 | 在哪跑 | 原因 |
|---|---|---|
| 量化校准 | 桌面卡 | 要加载 FP16 全模型 + 统计激活，3B 峰值 14-18GB，Orin 装不下 |
| engine build | **必须 Orin** | TensorRT engine 绑定 SM 架构、TRT 版本、cuDNN 版本 |

桌面卡（SM89）编的 engine 在 Orin（SM87）上 `deserializeCudaEngine` 直接返回
nullptr。这不是配置问题，是设计如此。

---

## 一、视觉塔导出

### 1.1 shape 契约（最容易错的地方）

```
pixel_values.shape == (t × h × w, 1176)
1176 = 3(RGB) × 2(temporal_patch) × 14 × 14

h % 2 == 0, w % 2 == 0                    ← 硬约束（merger 是 2×2）
n_visual_tokens = t × (h/2) × (w/2)
原图尺寸 = (h×14) × (w×14)，必须是 28 的整数倍
```

`export_vit_onnx.py` 里的 `grid_for_visual_tokens()` 会从目标 token 数反推
自洽的 (t,h,w)，并 assert 校验。**不要手算这三个数**，手算的错误率很高，
而且错了不报错、只是静默错位。

### 1.2 三个坑

**坑 1：cu_seqlens 被常量折叠**

ViT 内部用变长注意力，靠 cu_seqlens 分段。直接 `torch.onnx.export` 会把它
固化成常量，换张图就错。解法是 `VisionTowerWrapper` 把 grid_thw 作为显式输入。

**坑 2：动态 seq 维**

必须在 `dynamic_axes` 里声明 `pixel_values: {0: "n_patches"}`，
否则 engine 只能吃固定分辨率。

**坑 3：optimization profile 的 opt 档**

```bash
OPT_PATCHES=4096 bash convert/build_vit_engine.sh
```

TensorRT 按 opt 档做 kernel autotuning。设错了性能差一倍。
opt 应该是你**实际部署时最常见的分辨率**，不是最大值。

`export_vit_onnx.py` 会把契约写进 `vit_fp16.shapes.json`，
build 脚本直接读，避免两边手抄对不上。

### 1.3 验证

```bash
python convert/export_vit_onnx.py --opt-visual-tokens 1024
```

脚本会自动做 ONNX vs PyTorch 的余弦相似度校验。**< 0.999 就别往下走**，
排查方向脚本里打印了。

---

## 二、LLM 主干量化

见 `docs/03_quantization.md`。

产出 `ckpt/qwen25vl-3b-int4awq/`，含 `quant_recipe.json`
（记录了量化配置和校准集来源，面试要用，别删）。

---

## 三、Engine 构建

### 3.1 ViT

```bash
OPT_PATCHES=4096 MAX_PATCHES=16384 bash convert/build_vit_engine.sh
```

工作空间：Orin 是统一内存，`--memPoolSize=workspace` 别设太大。
8GB 板子 1024，16GB 板子 2048。

### 3.2 LLM

```bash
MAX_MM_LEN=4096 MAX_INPUT_LEN=6144 bash convert/build_llm_engine.sh
```

**三个参数的关系（算错就运行时报错）**

```
max_multimodal_len ≥ MAX_PATCHES / 4          # merger 2×2
max_input_len      ≥ max_multimodal_len + 文本 prompt 长度
max_seq_len        ≥ max_input_len + max_output_len
```

`max_multimodal_len` 会占 KV Cache 预算。设大了，可用 batch 变小。
这是 VLM 和纯语言模型的一个关键差异。

### 3.3 KV Cache 比例

```
kv_cache_free_gpu_memory_fraction
```

Orin 统一内存，这个 fraction 是相对**全部系统内存**，不是独立显存。

| 板子 | 建议值 | 要留给谁 |
|---|---|---|
| 8GB | 0.40 ~ 0.50 | ViT engine ~1.5GB、系统 ~1.2GB、图像预处理缓冲 |
| 16GB | 0.55 ~ 0.65 | 同上 |

### 3.4 版本兼容

TRT-LLM 的每个 tag 只支持特定 TensorRT 版本，JetPack 又绑定 TensorRT 版本。
三者必须对上。

**先跑 `benchmark/probe_device.py`，把输出发我，我确认该用哪个 tag。**
不要凭感觉 checkout，版本对不上就是浪费 4 小时编译。

Qwen2-VL 支持从 v0.12 起；Qwen2.5-VL 需要更新的 tag。
如果你的 JepPack 只能配到不支持 2.5-VL 的版本，退到 Qwen2-VL-2B——
架构同源，本仓代码兼容。

### 3.5 兜底方案

TRT-LLM 编不出来时：

- **ViT 走 TensorRT + LLM 走 llama.cpp(混合链路,已实测交付)**:手写 C++ 驱动(runtime/hybrid_driver.cpp,约170行,零改 llama.cpp 源码)经 llama_batch.embd 注入 ViT 特征,板上 TTFT≈1.88s(未含预处理与特征IO)对比纯 llama.cpp 链 P50 3.59s(含预处理)提速约 1.7~1.9 倍,decode 与纯链一致。数据:results/verified/orin/return_day2/hybrid_llamacpp_e2e.json,commit 296cd66。
- **MLC-LLM**：Jetson 支持较好，量化用 q4f16

**这条路不是兜底,是本项目的主交付。** 工程结论:定位 TRT-LLM 0.12 的 M-RoPE 兼容边界后,手写 C++ 驱动将 TensorRT ViT 特征注入 llama.cpp,混合链路使 VLM 首字延迟由 3.59 s 降至约 2 s(口径:1.88s 未含预处理,3.59s 含预处理)。
