# Benchmark 方法论

> 数据的可信度取决于测量方法。这份文档定义了每个指标的口径。
> **面试官问"你这个数字怎么测的"，答案在这里。**

## 零、测量前的固定动作

每次 benchmark 之前必须做，否则数据不可比：

```bash
sudo nvpmodel -m 0        # MAXN
sudo jetson_clocks        # 锁频，消除 DVFS 抖动
sleep 60                  # 等温度稳定
python benchmark/probe_device.py    # 记录环境
```

**`jetson_clocks` 每次重启后失效。** 这是"昨天还好好的今天变慢了"的头号原因。

三个必须记录的参数（面试第一个问的就是这个）：
- 板卡型号 + 内存
- JetPack / TensorRT / TRT-LLM 版本
- 功耗模式

---

## 一、指标定义

### 1.1 TTFT（首字延迟）

**VLM 必须拆开报，只报总数没有信息量。**

```
端到端 TTFT = 图像预处理 + ViT 编码 + prompt 拼接 + LLM prefill
```

拆开之后才能回答"你这 TTFT 里视觉编码占多少"——
这是判断优化方向的依据。本项目在纯 llama.cpp 链中实测视觉编码占大头,故第一优化落在视觉侧(ViT 迁移至 TensorRT,741ms);混合链路交付后 LLM prefill(1133ms)成为 TTFT 最大分项,下一步优化方向随之转移。

`bench_latency.py` 会自动分解并扫描不同分辨率。

### 1.2 decode 吞吐

```
tok/s = 输出 token 数 / (总耗时 - TTFT)
```

注意分母要减掉 TTFT，否则短输出时数字会被 prefill 严重稀释。

### 1.3 内存峰值

**Orin 是统一内存（UMA），没有独立显存。**

用 `/proc/meminfo` 的 `MemTotal - MemAvailable`，
或 `tegrastats` 的 RAM 字段。**不要用 `torch.cuda.max_memory_allocated()`**，
那只统计 PyTorch 分配的部分，漏掉 TensorRT engine 和系统占用。

报数时要说明是绝对占用还是相对基线的增量。

### 1.4 冷启动

从进程启动到第一次推理可用的时间，拆成：
- ViT engine 反序列化
- LLM engine 反序列化 + KV Cache 分配

车端关心这个是因为整车上电到功能可用有时间要求。

### 1.5 长稳与抖动

30 分钟连续压测，按 5 分钟分窗统计 P50/P95/P99。

**关键指标是漂移量**：末窗 P50 相对首窗 P50 的变化百分比。
这直接反映热降频的影响，是被动散热板子的真实问题。

同时记录结温曲线。>85°C 基本就在降频了。

### 1.6 功耗与能效比

从 `tegrastats` 读，梯形积分算能耗。

**必须说明电轨口径**：

| 电轨 | 含义 |
|---|---|
| `VDD_IN` | 整板输入功率（含 CPU/DRAM/外设/风扇） |
| `VDD_CPU_GPU_CV` | 计算单元功率 |
| `VDD_SOC` | SoC 其他部分 |

```
能效比 = 总输出 token 数 / 总能耗(J)   单位 tokens/J
```

**报能效比不写口径等于没报**，两个口径的数字差很多，不可直接比较。

注意 tegrastats 最小采样周期约 50ms，比单个 decode step 还长，
所以只能算平均功耗。**不要谎称测到了 per-token 功耗。**

### 1.7 并发吞吐

VLM 的并发瓶颈和纯语言不同：除了 KV Cache，还受 `max_multimodal_len` 约束。
`bench_throughput.py` 会扫并发档位找吞吐拐点，并记录失败请求数。

失败率突然上升通常不是 OOM，是超了 `max_multimodal_len`。

---

## 二、误差控制

### 2.1 预热

前几次调用包含 kernel autotuning、内存分配、CUDA context 初始化，
必须丢弃。本仓所有脚本默认预热 3~5 次。

### 2.2 重复与统计量

- 至少 20 次
- 报 **P50** 而不是均值（均值被尾部拖偏）
- 同时报 P95/P99，尾延迟才是车端真正关心的

### 2.3 控温

每轮测试之间 `sleep 120`。上一轮的余热会让下一轮起点温度更高，
数据不可比。`run_all.sh` 里已经加了。

### 2.4 隔离

测之前 `htop` 看一眼有没有其他进程。
Orin 的 CPU 弱，后台一个编译就能把数据搞乱。

---

## 三、Profiling

### 3.1 Nsight Systems（时间线）

```bash
nsys profile -o report python runtime/run_vl.py --image assets/demo.jpg
```

看哪一段是瓶颈：ViT / 桥接 / prefill / decode。

### 3.2 Nsight Compute（算子级）

```bash
ncu --set full --kernel-name fused_match_kernel python kernels/bench_kernel.py
```

关键指标：
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` —— 带宽利用率
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` —— 算力利用率
- `dram__bytes` —— 实际访存量（验证融合是否生效）

### 3.3 memory-bound 的判定

**两步，缺一不可：**

**① 理论分析**
```
decode 阶段每个 token 要读全部权重（W GB），计算量 2·params FLOPs
算术强度 ≈ 2 FLOP/byte
Orin ridge point = 算力 / 带宽
```

**ridge point 必须用实测带宽算**，不是 datasheet 标称值。
`probe_device.py` 会实测，Orin 实际可达通常是标称的 70~85%。

用标称值算出来的"带宽利用率"是假的，面试官一问就露。

**② 实测验证**
Nsight Compute 看 DRAM 吞吐达到峰值的百分比，同时看 SM 利用率。
memory-bound 的特征是前者高、后者低。

**注意 prefill 和 decode 结论相反**：prefill 是批量矩阵乘，compute-bound；
ViT 编码同理。所以两者的优化方向完全不同。

---

## 四、结果归档

```
results/
├── device_info.json        环境基线（先跑 probe_device.py）
├── raw/                    原始 json，可追溯
│   ├── latency_*.json
│   ├── throughput_*.json
│   ├── stability.json
│   ├── sensitivity.json
│   └── kernel_bench.json
├── figures/                自动出图
└── RESULTS_FILLED.md       自动填表
```

```bash
bash benchmark/run_all.sh              # 一键跑全套
python benchmark/plot_results.py       # 出图 + 填表
```

**填表脚本不会自动编数字。** 没跑的测试留空并标注缺哪个数据源——
自动编数比留空危险得多，你会不知道哪个数字是真的。
