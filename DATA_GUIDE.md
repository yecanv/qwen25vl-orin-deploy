# 数据说明

> 这个项目涉及两类完全不同的"数据"，得分开说，混在一起会误会。
>
> - **输入数据**：跑项目需要的模型、校准集、评测集、测试图 —— **全部随包提供或脚本自动获取**
> - **输出数据**：TTFT、吞吐、显存峰值等所有性能数字 —— **必须你自己在板子上跑出来**

---

# 第一部分：输入数据（要准备什么）

## 1. 模型权重

| 项 | 说明 |
|---|---|
| 模型 | `Qwen/Qwen2.5-VL-3B-Instruct`（16GB 板）/ `Qwen/Qwen2-VL-2B-Instruct`（8GB 板） |
| 体积 | 约 7 GB / 4.5 GB |
| 来源 | HuggingFace，开源可商用（Apache 2.0） |
| 费用 | 免费 |

**国内下载**：
```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen25vl-3b
```
或用 ModelScope（国内更快）：
```bash
pip install modelscope
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct --local_dir ./models/qwen25vl-3b
```

⏱ 约 30 分钟

---

## 2. 量化校准集 ★

**这是客户最关心的一块，全部是公开数据，脚本自动下载。**

```bash
python calib/build_vl_calib.py --probe          # 先探测哪些源可用
python calib/build_vl_calib.py --n-samples 512
```

### 2.1 数据源清单

| 数据集 | 用途 | 许可 | 默认权重 |
|---|---|---|---|
| COCO-Caption | 通用图像描述，分布最广 | CC BY 4.0 | 35% |
| TextVQA | 图中文字，激活分布特殊 | CC BY 4.0 | 30% |
| DocVQA | 文档类高分辨率 | 研究用途 | 20% |
| VQAv2 | 通用视觉问答 | CC BY 4.0 | 15% |

可选（需要本地 COCO 图片，用 `--coco-dir` 启用）：

| LLaVA-Instruct-150K | 视觉指令对齐 | 标注由 GPT-4 生成 | — |

> LLaVA-Instruct-150K **只有标注、不含图片**，图片要另外下 COCO train2017（19GB）。
> 脚本默认不启用它，只用自带图片的源，开箱即可跑。

### 2.2 校准集为什么必须是图文混合的

量化校准统计的是每层激活的动态范围。VLM 的 LLM 主干在推理时，
输入序列里很大一部分是**视觉 token**，激活分布和文本 embedding 完全不同。

直觉推断:只用纯文本校准 → scale 按文本分布定 → 视觉 token 被截断 → 看图胡说。

> ⚠️ **实测修正(桌面伪量化口径)**:上述直觉**未被复现**。
> {图文,纯文本}校准 × {图文,纯文本}探针 × α 四档全矩阵下,纯文本校准在
> 图文探针上并不更差(best-vs-best KL 0.0146 vs 0.0390)。机制:视觉巨激活
> 抬高 mean|x|,固定 α 下图文校准反而过保护。详见 results/verified/
> calib_ablation_desktop.json 与 docs/qbank/02 追问链。
> 本节的图文混合构建流程仍保留——它是做这组对照实验的工具。

脚本提供 `--text-only-ablation` 生成纯文本对照组(上述对照实验即用它构建)。

### 2.3 ⚠️ 体积注意

**这一条很容易踩坑**，脚本会在开始前提醒：

| max_pixels | 视觉 token | 单样本 | 128 条 | 512 条 |
|---|---|---|---|---|
| 200704 | 256 | 2.4 MB | 0.31 GB | 1.23 GB |
| **401408** | **512** | **4.8 MB** | **0.62 GB** | **2.47 GB** ← 推荐 |
| 802816（默认） | 1024 | 9.6 MB | 1.23 GB | **4.93 GB** |
| 1605632 | 2048 | 19.3 MB | 2.47 GB | 9.87 GB |

`.pt` 文件 `torch.load` 时要**全部读进内存**，太大会 OOM。

**推荐配置**：
```bash
python calib/build_vl_calib.py --n-samples 512 --max-pixels 401408
# → 约 2.5 GB，量化效果无损（校准分辨率不需要等于部署分辨率）
```

超过 6 GB 时脚本会警告并要求确认。

### 2.4 数量：多少条够

128 起步就能出效果（AWQ 原论文用 128 条），512 是精度/耗时的平衡点。
**建议做 128/256/512 的消融**，证明你知道边际收益在哪。

---

## 3. 评测集

```bash
python eval/build_eval_set.py --n 100
```

| 数据集 | 用途 | split |
|---|---|---|
| MMBench_CN | 中文多模态综合能力 | dev |
| TextVQA | 图中文字识别（量化最敏感） | **test**（校准用的是 validation） |

体积很小，100 条图片约 50 MB。

### ★ 与校准集的隔离验证

脚本会记录图片指纹（灰度缩放 16×16 后取 md5），
构建评测集时**自动剔除与校准集重叠的样本**，最后打印：

```
[eval] 因与校准集重叠而剔除 x 条
[eval] 无重叠，校准/评测分离成立 ✓
```

**这行输出面试要用。** 面试官问"你校准和评测是同一批数据吗"，
这就是证据。用评测集做校准是数据泄漏，一问就穿。

---

## 4. 测试图

```bash
python assets/make_demo.py
```

**程序化生成，不随包分发真实照片。** 两个原因：
- benchmark 需要可控的分辨率和内容复杂度
- 真实照片有版权问题

生成三档分辨率（448×448 / 896×896 / 1792×1008），
对应 `bench_latency.py` 的扫描档位。内容是模拟车载前视场景的合成图
（路面、车道线、建筑、车辆），带足够的高频细节来压视觉编码。

---

## 5. 磁盘空间预算 ★

| 项 | 体积 |
|---|---|
| Qwen2.5-VL-3B 权重 | 7.0 GB |
| ViT ONNX | 1.3 GB |
| TRT-LLM checkpoint（FP16 + INT4） | 9.2 GB |
| Engine（ViT + LLM FP16 + LLM INT4） | 10.7 GB |
| 校准集（fp16, 512 条 @401408） | 2.5 GB |
| 评测集 | 0.05 GB |
| TensorRT-LLM 源码 + 编译产物 | 25 GB |
| jetson-containers 镜像（走路线 A 时） | 15 GB |
| **合计** | **约 71 GB** |

**Orin 板载 eMMC 通常只有 16-64 GB，必须插 NVMe SSD，256 GB 起。**

这一条务必提前告诉客户，不然板子到手才发现装不下，白等一轮。

---

## 6. 国内网络

HuggingFace 直连很慢或不通，三个办法：

```bash
# 1. 镜像站（最简单）
export HF_ENDPOINT=https://hf-mirror.com

# 2. ModelScope（模型权重更快）
modelscope download --model Qwen/Qwen2.5-VL-3B-Instruct

# 3. 数据集拉不到时，脚本内置了备用源，会自动尝试
python calib/build_vl_calib.py --probe   # 先看哪些源可用
```

`--probe` 会逐个测试数据源并打印可用性，全不可用就是网络问题。

---

# 第二部分：输出数据（性能数字）

## ★ 这部分给不了，原因是技术性的

**所有 benchmark 数字必须在客户自己的板子上跑出来。** 不是流程规定，是三条硬约束：

**1. TensorRT engine 绑定硬件**
engine 里烧进了 SM 架构（Orin 是 sm_87）、TensorRT 版本、cuDNN 版本。
我在 4070Ti（sm_89）上 build 的 engine，Orin 上 `deserializeCudaEngine`
直接返回 nullptr。**这不是配置问题，是 kernel autotuning 在目标硬件实测的必然结果。**

**2. 同为 Orin 数据也不一样**
功耗模式（15W/25W/MAXN）、JetPack 版本、是否 `jetson_clocks`、
环境温度、散热方式——任何一项不同，延迟能差 30% 以上。

**3. 数字要能追溯才有用**
面试官问"你这个 TTFT 多少 ms、什么配置下测的"，
答不出板卡型号、JetPack 版本、功耗模式，前面讲的全废。

## 给了什么替代

| 提供 | 说明 |
|---|---|
| `results/RESULTS_TEMPLATE.md` | 完整数据表模板，8 张表 |
| `benchmark/run_all.sh` | 一键跑全套，约 2 小时 |
| `benchmark/plot_results.py` | 自动出图 + 自动填表 |
| `benchmark/probe_device.py` | 环境探测，输出即数据可信度凭证 |
| `docs/04_benchmark_methodology.md` | 每个指标的口径定义、误差控制 |

**填表脚本不会自动编数字。** 没跑的测试留空并标注缺哪个数据源——
自动编数比留空危险得多，你会不知道哪个数字是真的。

## 要产出哪些数据

跑完 `run_all.sh` 后，`results/` 下会有：

```
results/
├── device_info.json           环境基线（板卡/版本/实测带宽）
├── raw/
│   ├── latency_*.json         TTFT 分解、四档分辨率
│   ├── throughput_*.json      并发 1/2/4/8 的吞吐与尾延迟
│   ├── stability.json         30min 长稳、功耗、结温
│   ├── sensitivity.json       逐模块量化敏感度
│   ├── kernel_bench.json      融合算子 vs 朴素
│   └── consistency.json       量化前后精度回归
├── figures/                   6 张图
└── RESULTS_FILLED.md          自动填好的数据表
```

---

# 附：可以直接发给客户的话术

> **数据分两块说：**
>
> **要准备的输入数据，全部公开且免费，脚本自动下载。**
> 模型权重从 HuggingFace/ModelScope 拉（Qwen2.5-VL-3B，7GB，Apache 2.0 可商用）；
> 量化校准集用 COCO-Caption / TextVQA / DocVQA / VQAv2 混采，
> 脚本 `--probe` 先探测可用性再自动构建；
> 评测集用 MMBench_CN + TextVQA test，脚本会自动做与校准集的指纹去重；
> 测试图是程序化生成的，不涉及版权。
>
> **两个要提前准备的：**
> 1. **板子上必须插 NVMe SSD，256GB 起。** 模型 + 编译产物 + engine 加起来约 70GB，
>    板载 eMMC 装不下。
> 2. 校准集建议用 `--max-pixels 401408`，文件约 2.5GB；用默认值会到 5GB，
>    加载时占内存。
>
> **性能数字给不了，必须你自己跑。** 不是我藏着——TensorRT engine 绑定
> SM 架构和 TRT 版本，我机器上编的你板子加载不了；
> 就算同为 Orin，功耗模式、JetPack 版本、散热条件不同，延迟能差 30%。
>
> 我给的是数据表模板 + 一键跑测脚本 + 自动出图填表 + 每个指标的口径定义。
> 你跑一遍 `run_all.sh`（约 2 小时），数据就自动填进表里。
>
> 这样你简历上每个数字都是自己板子跑出来的，
> 面试官问"什么配置测的"你答得上——这比拿一份别人的数据强太多。
