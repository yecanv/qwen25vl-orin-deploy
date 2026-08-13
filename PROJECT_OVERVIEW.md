# Qwen2.5-VL 车端多模态推理部署与量化压缩

**项目描述文档**

---

> ## ⚠️ 状态声明（务必先读）
>
> **本项目已上板实测交付:混合链路(TensorRT ViT 引擎 + 手写 C++ 驱动经 llama_batch.embd 注入 llama.cpp)端到端板上实测 PASS(TTFT≈1.88s,commit 296cd66);TRT-LLM 0.12 栈内端到端仍不通(无 M-RoPE)。三条端到端路线定案:①外置 RoPE 补 0.12(周级)②JP7+Edge-LLM(半天官方,可选)③混合链路(已交付)。**
>
> | 已完成 | 未完成 |
> |---|---|
> | 架构设计与技术方案;板上实测:混合链路端到端 PASS、ViT FP16 引擎 741ms、llama.cpp 纯链基线 3.59s | TRT-LLM 0.12 栈内端到端(无 M-RoPE,仍不通)、多图单请求驱动、外置 RoPE 补 0.12(周级路线) |
> | 全部代码与文档 | TensorRT-LLM / ModelOpt 的 API 适配 |
> | M-RoPE 逻辑离线验证 | engine 编译与端到端链路 |
> | CUDA 算子算法验证（CPU 对拍） | CUDA kernel 的实际编译与性能 |
核心性能数字已板上实测并入账 results/verified/（DATA_LEDGER 可溯源）；未跑的表项仍留空
>
> 下文描述的是**技术方案与设计意图**，不是已完成的实验结论。
> 凡涉及"扫描结果""对照实验结论""实测数字"的表述，
> 均指**该实验的设计方案**，实际数据需在目标硬件上产出。
>
> **简历与面试中,已实测部分按已交付口径讲(以简历 v2 行文为准);仍未实测部分(多图单请求驱动、外置 RoPE 路线、Nsight 剖析等)按进行中口径,不讲结论。**
> 把设计意图说成已完成的实验，在自动驾驶/机器人公司是一票否决的问题——
> 定性是造假，不是"深度不够"。

---

## 一句话

把 Qwen2.5-VL 多模态大模型压进 Jetson Orin 车规级平台，
打通视觉编码、量化压缩、推理服务、性能评测的全链路，
并建立一套可追溯的车端性能评测体系。

---

## 一、项目背景

多模态大模型在自动驾驶的场景理解、端到端决策上有明确价值，
但从云端搬到车端会撞上四堵墙：

| 约束 | 云端 | 车端（Orin） |
|---|---|---|
| 显存 | 80 GB 独显 | 8/16 GB **统一内存**，与系统共用 |
| 带宽 | 3 TB/s (HBM) | ~102 GB/s (LPDDR5) |
| 功耗 | 400 W | 15~25 W，有功耗墙 |
| 散热 | 主动强制风冷 | 多为被动散热，跑久了热降频 |
| 实时性 | 尽力而为 | 有硬性延迟预算 |

**桌面卡上跑得动，不代表车端跑得动。** 这个项目要回答的就是：
差距具体在哪、能不能补上、代价是什么。

---

## 二、技术目标

| 维度 | 目标 |
|---|---|
| 模型 | Qwen2.5-VL-3B-Instruct（16GB 板）/ Qwen2-VL-2B（8GB 板） |
| 硬件 | Jetson Orin NX 16GB / Orin Nano Super 8GB |
| 引擎 | TensorRT（视觉塔）+ TensorRT-LLM（语言主干） |
| 精度 | ViT 保 FP16，LLM 主干 INT8-SmoothQuant / INT4-AWQ |
| 评测 | TTFT、吞吐、显存峰值、冷启动、长稳抖动、功耗能效、精度回归 |

---

## 三、系统架构

```
                  ┌──────────────────────────────────────────┐
   image ───────► │ 视觉塔 ViT（FP16，TensorRT engine）        │
                  │  动态分辨率 → patch embed → window attn   │
                  │  → 2D-RoPE → patch merger (2×2)          │
                  └──────────────┬───────────────────────────┘
                                 │ vision embeds [N_vis, hidden]
                                 ▼
                  ┌──────────────────────────────────────────┐
                  │ 视觉 token 压缩（手写 CUDA 融合算子）       │
                  │  双向图软匹配 → top-r 合并                 │
                  │  归一化+相似度+argmax 三合一，S 不落地      │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
   text  ───────► │ 桥接层                                    │
   tokens         │  prompt table：fake_id = vocab_size + idx │
                  │  M-RoPE：(t,h,w) 三维位置编码展开          │
                  └──────────────┬───────────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────────┐
                  │ 语言主干（INT8-SQ / INT4-AWQ）             │
                  │  TensorRT-LLM + Paged KV Cache           │
                  │  inflight batching + 流式输出             │
                  └──────────────┬───────────────────────────┘
                                 ▼
                            streaming output
```

---

## 四、四个核心技术点

### 4.1 双引擎异构与 M-RoPE 对齐

**为什么分两个引擎**：ViT 是定长前馈、compute-bound；LLM 是自回归、
decode 阶段 memory-bound。合成一个 engine 无法分别优化。分开后还有个好处：
同一张图多轮问答时视觉编码只跑一次，后续复用。

**最难的地方是 M-RoPE**。Qwen2-VL 系列不用标准 1D RoPE，而是把位置拆成
(temporal, height, width) 三分量。图像 token 的 h/w 按 patch 在二维网格的
坐标展开，而**图像之后的文本 token，位置不是累加视觉 token 数，
而是接在 max(h, w) 之后**。

这行写错不报错，只会让输出慢慢退化——先是细节描述错乱，再是长文本重复。
`runtime/mrope.py` 实现了这个逻辑，并**已用构造样例离线验证**
（图后文本位置、prompt table fake id 映射均正确）；
`selftest_against_hf()` 提供与 HuggingFace 参考实现的逐元素对拍，
**需在有模型权重的环境下运行**。

### 4.2 混合精度量化决策

**方案：不全量化，而是基于敏感度分析做选择性量化。**

设计的验证方法是逐模块量化敏感度扫描——统计每个 Linear 层输入激活的 outlier ratio
（per-channel absmax 的 max/median），并对每组模块做 INT8 伪量化后测输出
logits 的 KL 散度。

**预期结论**（待实测验证）：ViT 的 outlier channel 位置随输入图像漂移，
静态的 per-channel scale 覆盖不住；而 ViT 只占参数量约 18%
（约 0.67B / 总 3.75B）、且只在 prefill 跑一次。
因此方案定为**保 FP16，只量化 LLM 主干**。

`quantize/sensitivity_analysis.py` 会产出散点图（X=参数占比即收益，
Y=KL 散度即代价）。**该决策现已有双重支撑:敏感度分析之外,ViT INT8 实战六墙(FP32校准契约→DDS杀执行器→ModelOpt批次→fp16 inf 卡熵→bin退化→构建病态慢)证明端侧连构建量化 ViT 引擎都不划算,反向加固保 FP16。**

### 4.3 多模态校准集设计

**这是项目里最有独立发现价值的一条。**

量化校准统计的是每层激活的动态范围。VLM 的 LLM 主干在推理时，
输入序列里很大一部分是视觉 token——由 ViT 输出、经 projector 投影而来，
激活分布和文本 embedding 完全不同。

只用纯文本语料校准，量化 scale 按文本分布定，实际推理时视觉 token
一进来就被大面积截断。**现象是：文字问答正常，一看图就胡说。**

`calib/build_vl_calib.py` 构建图文混合校准集（COCO-Caption / TextVQA /
DocVQA / VQAv2 混采），并提供 `--text-only-ablation` 生成纯文本对照组。

**对照实验已于 2026-08-05 完成(桌面伪量化口径),直觉未被复现**——纯文本校准在图文探针上并不更差,机制为视觉巨激活抬高 mean|x| 导致固定 α 下图文校准过保护(results/verified/calib_ablation_desktop.json);
跑之前只是一个基于激活分布差异的合理推断。

校准集/评测集隔离机制已实现（图片指纹去重），同样需实跑验证。

### 4.4 视觉 token 压缩融合算子

**设计动机来自瓶颈分析（该分析需实测确认）。**

高分辨率输入下视觉 token 占 prompt 的绝大部分：
1024×1024 的图，预处理 smart_resize 先把边长按 28 像素对齐
（round(1024/28)=37 → 1036），patch(14) 后每边 74，merger(2×2) 后
37² = **1369 个视觉 token**，而文本 prompt 通常只有几十个。这直接主导两件事：
prefill 的 attention 是 O(n²) 决定 TTFT；KV Cache 占用决定可用并发。

算法用双向图软匹配（按下标奇偶拆分 A/B 两组，为每个 a 找最相似的 b，
合并相似度最高的 r 对）。

**融合的动机**：朴素实现要三次 kernel launch——L2 归一化 → 相似度矩阵
S = Â·B̂ᵀ → 逐行 argmax。而 S 是纯中间产物，我们只要每行的 max 和 argmax。
融合后 S 从不落地：寄存器里累加完一个 tile 就立刻并入 running max。

实现细节：分块 32×32×32、256 线程；线程映射用**跨步取列**
（n = tid%8 + 8k），让同一行的 8 个线程落在 warp 内连续 lane，
行内归约直接走 `__shfl_down_sync(width=8)` 不经过 shared memory；
shared memory 数组 pad 到 33 避开 bank conflict。

**收益要分两层说**：kernel 级省的是访存和 launch 开销（GEMM 的算力开销
一分没省）；端到端的大头在 token 数减少后 attention 的 O(n²) 降维——
压缩 25% 时 attention 计算量降到 56%，压缩 50% 时降到 25%（这一层是纯数学）。

**kernel 级的实际加速比未测**，需在 Orin 上跑 `kernels/bench_kernel.py`。
算法逻辑已用 CPU 镜像对拍验证（8 组边界用例，argmax 完全一致）。

---

## 五、评测体系

面向车端场景设计，覆盖云端评测通常不看的指标：

| 指标 | 说明 |
|---|---|
| **TTFT（分解）** | 拆成视觉编码 / LLM prefill 两段，只报总数没有信息量 |
| decode 吞吐 | tok/s，分母扣掉 TTFT |
| **显存峰值** | Orin 是统一内存，用 /proc/meminfo 而非 torch.cuda |
| **冷启动** | 整车上电到功能可用有时间要求 |
| 并发吞吐 | 扫并发档位找拐点；VLM 的瓶颈常是 max_multimodal_len 而非 KV Cache |
| **长稳与抖动** | 30min 连续压测，分窗统计 P50/P95/P99 漂移 + 结温曲线 |
| **功耗与能效** | tegrastats 梯形积分，tokens/J，必须注明电轨口径 |
| 精度回归 | 逐 token 一致性 + 任务准确率 + logits KL |

配套 Nsight Systems（时间线）与 Nsight Compute（算子级）分析流程，
用于判定 decode 阶段是否为 memory-bound——**ridge point 要用实测带宽算，
不用 datasheet 标称值**，因为 Orin 实际可达带宽通常只有标称的 70~85%。

`benchmark/probe_device.py` 会实测带宽。**该分析尚未执行。**

---

## 六、工程量

```
37 个文件，约 6400 行

├── docs/          6 份技术文档（环境踩坑、模型转换、量化、
│                  Benchmark 方法论、面试问答、简历写法）
├── env/           Jetson 环境配置 + TRT-LLM 编译（含兜底方案）
├── calib/         多模态校准集构建（含数据源探测、配额重分配、指纹去重）
├── convert/       ViT ONNX 导出、LLM 量化、双 engine 构建
├── quantize/      逐模块量化敏感度分析
├── runtime/       M-RoPE 实现 + 端到端推理链路 + token 压缩集成
├── kernels/       CUDA 融合算子 + CPU 镜像验证 + 绑定 + benchmark
├── benchmark/     设备探测、延迟分解、并发吞吐、功耗长稳、出图填表
├── eval/          评测集构建（校准隔离验证）、精度回归
├── assets/        测试图程序化生成
└── results/       数据表模板 + 自动填充
```

---

## 七、可复现性

**所有性能数字由使用者在自己的板卡上跑出，不随包分发。**

原因是技术性的：TensorRT engine 绑定 SM 架构、TensorRT 版本、cuDNN 版本，
在别的机器上 build 的 engine 加载会直接失败。即便同为 Orin，
功耗模式、JetPack 版本、是否 jetson_clocks、环境温度都能让延迟差出 30% 以上。

配套提供：
- 环境探测脚本（记录板卡型号、版本栈、实测带宽），输出即数据的可信度凭证
- 一键跑测脚本，结果写入 `results/raw/` 可追溯
- 自动出图与填表，**没跑的测试留空并标注缺哪个数据源，不自动编数字**

---

## 八、已离线验证的部分

| 模块 | 验证方式 | 结果 |
|---|---|---|
| M-RoPE 位置编码 | 构造样例逐元素校验 | 图后文本位置、fake id 映射均正确 |
| ViT shape 推导 | 7 组 token 数 + 非方形长宽比 | 全部自洽 |
| 校准集配额分配 | 5 种数据源失败组合 | 总数均达标 |
| CUDA 融合算子 | CPU 镜像 vs 朴素实现，8 组用例含边界 | argmax 完全一致，误差 < 6e-7 |
| 全量语法 | Python + Shell + C++ | 全部通过 |

**未验证**:TensorRT-LLM / ModelOpt 的具体 API 签名(版本漂移大,需按实际版本适配)、CUDA 融合算子的板上编译与性能。GPU 端到端已由混合链路实测交付(TRT ViT 741ms + hybrid_driver.cpp 注入 llama.cpp,TTFT≈1.88s,results/verified/orin/return_day2/);TRT-LLM 0.12 栈内端到端仍不通(无 M-RoPE)。
