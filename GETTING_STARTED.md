# 保姆级上手教程

> **读这一份就够了。** 每一步都有：做什么 → 预计耗时 → **验收标准**。
> 验收不通过就别往下走，往下走只会在更远的地方以更难懂的方式崩。
>
> 卡住了看 `docs/01_setup_troubleshooting.md`，它按**报错信息**索引。

---

## 开始之前：你需要两台机器

这是整个流程最容易搞混的地方，先说清楚。

| | 机器 A：你的台式机 | 机器 B：Jetson Orin |
|---|---|---|
| 配置 | RTX 4070Ti SUPER（你现有的） | Orin NX 16GB / Orin Nano 8GB |
| 干什么 | 下模型、导 ONNX、**量化校准** | **编译 engine、跑推理、所有 benchmark** |
| 为什么 | 量化要加载 FP16 全模型统计激活，3B 峰值 14-18GB，Orin 装不下 | TensorRT engine 绑定 SM 架构，必须在目标板上 build |

**产物流转方向：**

```
机器 A                                    机器 B
──────                                    ──────
下载模型 ──┐
           ├─► onnx/vit_fp16.onnx ────────► build ViT engine
导出 ONNX ─┘
                                          
校准+量化 ──► ckpt/xxx-int4awq/ ──────────► build LLM engine
                                                 │
                                                 ▼
                                          跑推理 + benchmark
                                                 │
                                                 ▼
                                          results/（你的数据）
```

**engine 文件永远不要从 A 拷到 B。** 拷过去也加载不了。

---

## 第 0 天：买板子和装系统

### 0.1 选板子

| 板子 | 价格 | 能跑 | 建议 |
|---|---|---|---|
| **Orin NX 16GB** | ~4000 | Qwen2.5-VL-3B 全流程 | **推荐**，数据好看，不憋屈 |
| Orin Nano Super 8GB | ~1700 | Qwen2-VL-2B | 预算紧的选择，会比较紧 |

买套件版（含载板、散热、电源），别买光模组。
闲鱼租一个月 200-300 也行。

**顺便买**：一张 NVMe SSD（256GB 起，模型和编译产物很吃空间）、
一个 USB 转 TTL 串口线（刷机和救砖用得上）。

### 0.2 刷 JetPack

用 NVIDIA SDK Manager，在 Ubuntu 主机上刷。
**建议刷 JetPack 6.x**（L4T 36.x），TensorRT 版本较新。

⏱ 约 2 小时（下载占大头）

### ✅ 验收 0

板子能开机进系统，`ssh` 能连上，然后：

```bash
cat /proc/device-tree/model
# 期望看到：NVIDIA Jetson Orin NX Engineering Reference Developer Kit 之类

nvidia-smi        # Jetson 上可能没有，正常
tegrastats        # 这个必须有，Ctrl+C 退出
```

`tegrastats` 跑不起来 = 系统没装好，重刷。

---

## 第 1 天：环境准备

### 1.1 【机器 B】跑环境配置脚本

```bash
git clone <你的仓库>   # 或者解压我给的包
cd qwen25vl-orin-deploy
sudo bash env/jetson_setup.sh
```

脚本会做：设 MAXN 功耗模式、锁频、建 16G swap、关 zram、
问你要不要关图形界面（**8GB 板子一定要关**，省 1GB 内存）。

⏱ 约 20 分钟

### 1.2 【机器 B】记录环境基线

```bash
python3 benchmark/probe_device.py
```

**这一步的输出你要背下来**，面试第一个问的就是这三个：
板卡型号、JetPack 版本、功耗模式。

### ✅ 验收 1.2

```
输出里应该有：
  "model": "...Orin..."                    ← 确认是 Orin
  "sm": "8.7"                              ← 必须是 8.7
  "measured_bandwidth_gbs": 70~90          ← 实测带宽，标称 102 的 70-85%
  "nvpmodel": 包含 MAXN                    ← 不是 MAXN 的话性能数据不可比
  "swap_total_mb": >= 16000                ← 8GB 板子必须够
```

任何一条不对，回去看 `docs/01_setup_troubleshooting.md` 第二节。

### 1.3 【机器 B】装 TensorRT-LLM

**⚠️ 这一步是整个项目最容易卡住的地方，也最耗时。**

**先别急着编译。** 把上一步 `probe_device.py` 的输出发我，
我确认你这个 JetPack 版本该用 TRT-LLM 的哪个 tag。
**版本对不上就是浪费 4 小时。**

确认后：

```bash
bash env/build_trtllm.sh
# 选 A：用 jetson-containers 预编译镜像（省 4 小时，推荐先试）
# 选 B：源码编译（A 不行时的兜底）
```

⏱ 路线 A 约 1 小时（主要是拉 15GB 镜像）；路线 B 约 3-6 小时

### ✅ 验收 1.3

```bash
python3 -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
python3 -c "import tensorrt; print(tensorrt.__version__)"
```

两个都能打印版本号才算过。

**如果编译反复失败**:别死磕。改走 ViT-TensorRT + LLM-llama.cpp 组合,见 `docs/02_model_conversion.md` 3.5 节。
这个组合就是本项目后来实测交付的混合链路(hybrid_driver.cpp 经 llama_batch.embd 注入,板上 TTFT≈1.88s 未含预处理 vs 纯链 3.59s 含预处理,decode 持平)——不是性能降级,是三条端到端路线中已交付的一条。
面试讲"我做了 A/B 对比后选了 X"比"照教程跑通了"更好。

### 1.4 【机器 A】装依赖

```bash
pip install -r requirements.txt
```

⏱ 10 分钟

### ✅ 验收 1.4

```bash
python -c "import torch, transformers, modelopt; print('ok')"
python -c "from transformers import AutoProcessor; print('ok')"
```

---

## 第 2 天：先把 FP16 跑通

> **重要原则：先能跑 → 再能测 → 再优化。**
> 很多人上来就量化，跑不通不知道是量化的问题还是链路的问题。

### 2.1 【机器 A】下模型

```bash
# 国内建议用镜像
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct --local-dir ./models/qwen25vl-3b
```

⏱ 约 30 分钟（7GB）

**8GB 板子的话下 `Qwen/Qwen2-VL-2B-Instruct`**
（Qwen2.5-VL 最小是 3B，没有 2B）。

### 2.2 【机器 A】M-RoPE 自检

**这一步别跳过。** 位置编码错了，后面所有输出都是错的，
而且不报错、很难查。

```bash
python runtime/mrope.py
```

### ✅ 验收 2.2

```
position_ids 一致: True
mrope_delta  ours=-2  ref=-2
自检通过
```

**看到 `False` 就停下来**，把输出发我。往下走没有意义。

### 2.3 【机器 A】导出 ViT ONNX

```bash
python convert/export_vit_onnx.py \
    --model ./models/qwen25vl-3b \
    --opt-visual-tokens 1024
```

⏱ 约 10 分钟

### ✅ 验收 2.3

```
[vit] dummy 输入规格（已校验自洽）:
      grid_thw     = (1, 64, 64)
      n_patches    = 4096
      视觉 token   = 1024
      等效原图     = 896x896
...
[vit] 余弦相似度 = 0.999xxx
[vit] OK
```

**余弦相似度 < 0.999 就别往下走。** 排查方向脚本里打印了。

记下最后一行提示的 `OPT_PATCHES=4096`，下一步要用。

### 2.4 拷贝到机器 B

```bash
scp -r onnx/ orin@<板子IP>:~/qwen25vl-orin-deploy/
```

### 2.5 【机器 B】构建 ViT engine

```bash
OPT_PATCHES=4096 MAX_PATCHES=16384 bash convert/build_vit_engine.sh
```

⏱ 约 20-40 分钟（TensorRT 要在板子上实测各种 kernel）

### ✅ 验收 2.5

```bash
ls -lh engines/vit_fp16.engine     # 应该有几百 MB
cat engines/vit_fp16.engine.buildinfo   # 记录了编译环境，别删
```

### 2.6 【机器 B】构建 FP16 的 LLM engine（不量化）

先跑通不量化的版本，作为基线和对照组。

> ⚠️ **待补**：`convert/hf_to_trtllm.py` 尚未提供（FP16 基线转换脚本缺失），
> 需补写该脚本或给 `convert/quantize_llm.py` 增加 fp16 直通模式后，本步骤才可执行。

```bash
# 把 HF 模型转成 TRT-LLM checkpoint（不量化）
python convert/hf_to_trtllm.py --model ./models/qwen25vl-3b --out ckpt/fp16

MAX_MM_LEN=4096 bash convert/build_llm_engine.sh ckpt/fp16 engines/llm_fp16
```

⏱ 约 30-60 分钟

### 2.7 【机器 B】第一次跑推理

```bash
python3 assets/make_demo.py        # 生成测试图
python3 runtime/run_vl.py \
    --llm-engine engines/llm_fp16 \
    --image assets/demo.jpg \
    --prompt "详细描述这张图片。"
```

### ✅ 验收 2.7 —— **这是整个项目最关键的验收点**

```
输出应该是一段通顺的、和图片内容相符的中文描述。

然后是指标：
n_visual_tokens      1024
e2e_ttft_ms          xxx
decode_tok_s         xx.x
```

**三种失败情况的区分（很重要）：**

| 现象 | 原因 | 去看 |
|---|---|---|
| 输出乱码、无意义字符 | M-RoPE 错了 | 回 2.2 重跑自检 |
| 说人话但和图无关 | prompt table 没接上 | docs/01 第四节 |
| 前面正常，十几个 token 后重复 | decode 忘加 mrope_delta | runtime/mrope.py |

**跑通这一步，项目就成了一半。** 剩下的都是在这条链路上做优化。

---

## 第 3 天：量化

### 3.1 【机器 A】探测校准集数据源

```bash
python calib/build_vl_calib.py --probe
```

### ✅ 验收 3.1

至少 2 个源显示"可用"。全不可用的话设镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 3.2 【机器 A】构建校准集

```bash
python calib/build_vl_calib.py --n-samples 512
```

⏱ 约 30 分钟

### ✅ 验收 3.2

```
[calib] 收集完成 512 条          ← 必须是 512，少了会打印警告
[calib] 统计：
        by_source  {'textvqa': xxx, 'coco_caption': xxx, ...}
```

**如果显示"只收集到 XXX/512"**，看它打印的失败源，
按提示用 `--sources` 只用可用的源重跑。

### 3.3 【机器 A】量化敏感度分析

这一步产出的是"为什么 ViT 不量化"的证据，**面试要用**。

```bash
python quantize/sensitivity_analysis.py --model ./models/qwen25vl-3b
```

⏱ 约 20 分钟

### ✅ 验收 3.3

```
  vision_patch_embed     参数占比  x.xx%  outlier(max)  xxx.xx  KL x.xxxx
  vision_attn            ...
  llm_mlp_gate_up        ...
```

**你要能看出这张表在说什么**：vision 那几行的 outlier ratio 和 KL
应该明显高于 llm 那几行。这就是 ViT 保 FP16 的依据。

看不出差异的话，把 json 发我看看。

### 3.4 【机器 A】量化

```bash
python convert/quantize_llm.py \
    --model ./models/qwen25vl-3b \
    --qformat int4_awq \
    --calib calib/data/vl_calib_512.pt \
    --out ckpt/int4awq
```

⏱ 约 1-2 小时（校准 loop 慢）

### ✅ 验收 3.4

```bash
ls ckpt/int4awq/                    # 有 rank0.safetensors + config.json
cat ckpt/int4awq/quant_recipe.json  # 记录了量化配置和校准集来源，别删
```

检查 `quant_recipe.json` 里的 `excluded_modules` **包含 visual 相关的项**。
不包含的话说明 ViT 被量化了，回去看 `docs/03_quantization.md` 2.1 节。

### 3.5 拷到机器 B 并 build

```bash
scp -r ckpt/int4awq orin@<板子IP>:~/qwen25vl-orin-deploy/ckpt/
# 机器 B 上：
MAX_MM_LEN=4096 bash convert/build_llm_engine.sh ckpt/int4awq engines/llm_int4awq
```

### 3.6 【机器 B】跑量化版

```bash
python3 runtime/run_vl.py --llm-engine engines/llm_int4awq --image assets/demo.jpg
```

### ✅ 验收 3.6

输出应该仍然通顺、和图相符。对比 FP16 版：
- decode_tok_s 应该**明显提高**
- 内存占用应该**明显下降**

**如果一看图就胡说** → 校准集问题，你可能用了纯文本校准。
**如果 OCR 类问题答不对** → ViT 被量化了，回 3.4 检查 excluded_modules。

### 3.7 【可选但强烈建议】对照实验

这是项目里最有独立发现价值的一条，**面试的加分项**：

```bash
# 机器 A：用纯文本校准集量化一版
python calib/build_vl_calib.py --text-only-ablation --n-samples 512
python convert/quantize_llm.py --calib calib/data/text_only_512.pt --out ckpt/int4awq_textonly
# 拷到 B、build、跑，对比图文任务的表现
```

如果纯文本版的图文任务明显掉点，你就**用数据证明了一个论点**，
而不是复述别人的结论。

---

## 第 4 天：CUDA 算子

### 4.1 【任意机器】先跑 CPU 镜像验证

不需要 GPU，先确认算法对：

```bash
cd kernels/ref
g++ -O2 -std=c++17 token_merge_ref.cpp -o ref && ./ref
```

### ✅ 验收 4.1

```
N=   64 D=   64 | argmax不一致   0 | ... | PASS
...
全部通过
```

### 4.2 【机器 B】编译 CUDA 扩展

```bash
cd kernels
CUDA_ARCH=87 python3 setup.py install
```

⏱ 约 10 分钟

### ✅ 验收 4.2

```bash
python3 -c "import token_merge_cuda; print('ok')"
```

### 4.3 【机器 B】算子 benchmark

```bash
cd ..
python3 kernels/bench_kernel.py --sweep
```

### ✅ 验收 4.3

```
      N      D   融合(ms)   朴素(ms)     加速    S矩阵  argmax一致
    256   2048     x.xxxx     x.xxxx   x.xxx    0.13M     100.0%
   1024   2048     ...
```

**`argmax一致` 必须是 100%。** 不是的话算法有问题，发我看。

**加速比不用追求好看。** 记住这个数字的意义：省的是访存和 launch，
不是算力。规模小时 launch 占比高，规模大时 S 矩阵访存占比高。

### 4.4 【机器 B】端到端压缩测试

```bash
python3 kernels/bench_kernel.py --e2e
```

这会扫 keep_ratio = 1.0 / 0.9 / 0.75 / 0.6 / 0.5，
看 TTFT 怎么降。**这才是真正的收益。**

---

## 第 5 天：全套 Benchmark

### 5.1 【机器 B】一键跑全套

```bash
bash benchmark/run_all.sh
```

⏱ **约 2 小时**（含 30 分钟长稳 + 各轮之间的降温）

**期间不要用这台板子做别的事**，后台跑东西会污染数据。

脚本会依次跑：环境基线 → 延迟分解 → 并发吞吐 → 敏感度 → 30min 长稳 → 出图填表。

### ✅ 验收 5.1

```bash
ls results/raw/       # 应该有 latency_*.json, throughput_*.json, stability.json 等
ls results/figures/   # 应该有 6 张 png
cat results/RESULTS_FILLED.md
```

**`RESULTS_FILLED.md` 开头会列出"缺失的数据源"**，
按提示补跑对应脚本。

### 5.2 【机器 B】Nsight Profiling

面试问"你怎么知道是 memory-bound"，答案要靠这个。

```bash
# 时间线
nsys profile -o report python3 runtime/run_vl.py --image assets/demo.jpg

# 算子级
ncu --set full --kernel-name fused_match_kernel python3 kernels/bench_kernel.py
```

**要记下来的两个数**：
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` —— 带宽利用率
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` —— 算力利用率

memory-bound 的特征是前者高、后者低。

### 5.3 精度回归

```bash
python3 eval/build_eval_set.py --n 100
python3 eval/eval_consistency.py
```

### ✅ 验收 5.3

```
[eval] 因与校准集重叠而剔除 x 条
[eval] 无重叠，校准/评测分离成立 ✓
```

**这行输出面试要用**，是"校准和评测没用同一批数据"的证据。

---

## 第 6 天：整理与面试准备

### 6.1 填数据表

打开 `results/RESULTS_FILLED.md`，把自动没填上的手工补齐。

**空着的格子不要编数字。** 没跑的测试就留空，
或者回去把对应的脚本跑了。

### 6.2 写简历

打开 `RESUME_GUIDE.md`，用版本 B 或 C，
把 `___` 全部替换成你自己的数字。

**填不上的整句删掉。**

### 6.3 过面试题

1. 先过 `docs/05_interview_qa.md` —— 针对这个项目的追问
2. 再过 `docs/qbank/` —— 43 题定向题库，带 ★ 的 30 题必须能答

### ✅ 最终验收

`RESUME_GUIDE.md` 最后有三层自检清单，逐条打勾。

**打不满就别投。** 尤其这几条：

- [ ] 板卡型号、JepPack 版本、功耗模式（面试第一个问的）
- [ ] 端到端 TTFT，视觉编码占多少
- [ ] 为什么 ViT 不量化，数据是什么
- [ ] M-RoPE 图后文本位置怎么算，算错什么现象
- [ ] CUDA 算子省的是访存不是算力
- [ ] token 压缩的两层收益

---

## 总耗时预估

| 阶段 | 耗时 | 说明 |
|---|---|---|
| 第 0 天 准备 | 半天 | 刷机为主 |
| 第 1 天 环境 | **1-2 天** | TRT-LLM 编译是最大变数 |
| 第 2 天 FP16 跑通 | 1 天 | 最关键的一天 |
| 第 3 天 量化 | 1 天 | 校准 loop 慢，可以挂着跑 |
| 第 4 天 CUDA 算子 | 半天 | |
| 第 5 天 Benchmark | 半天 | 跑测 2 小时，其余等着 |
| 第 6 天 整理 | 1 天 | |

**顺利的话一周，卡在 TRT-LLM 编译上可能两周。**
把时间预留出来，别等到投递前一周才开始。

---

## 卡住了怎么办

**先自己查**：`docs/01_setup_troubleshooting.md` 按报错信息索引，
覆盖了内存、engine、输出不对、性能、编译五类常见问题。

**发我的话，带这四样**（少一样我就得来回问）：

1. 完整报错栈（不是截图最后一行）
2. `python3 benchmark/probe_device.py` 的输出
3. 对应的 build log（`logs/` 目录下）
4. 你执行的完整命令

有这四样，大部分问题我能直接定位。

---

## 三条容易被忽略的提醒

**一、`jetson_clocks` 重启后失效。**
这是"昨天还好好的今天变慢了"的头号原因。
每次重启后、每次 benchmark 前都要重跑 `sudo jetson_clocks`。

**二、benchmark 之间要降温。**
上一轮的余热会让下一轮起点温度更高，数据不可比。
`run_all.sh` 里已经加了 sleep，手动跑的时候别省。

**三、engine 不要跨机器拷。**
这条说三遍都不多。拷过去加载失败，
而且报错信息（`deserializeCudaEngine` 返回 nullptr）完全指不到根因。
