# 文档索引

## 项目文档

| 文件 | 内容 |
|---|---|
| [../PROJECT_OVERVIEW.md](../PROJECT_OVERVIEW.md) | **项目描述**：背景、架构、四个核心技术点、评测体系 |
| [../README.md](../README.md) | 快速开始、目录结构、硬件适配矩阵 |
| [../DELIVERY.md](../DELIVERY.md) | 交付清单与验证状态 |

## 操作文档

| 文件 | 内容 |
|---|---|
| [01_setup_troubleshooting.md](01_setup_troubleshooting.md) | 环境搭建与踩坑排查（**按报错信息索引**） |
| [02_model_conversion.md](02_model_conversion.md) | 模型转换与 engine 构建 |
| [03_quantization.md](03_quantization.md) | 量化流程与校准集设计 |
| [04_benchmark_methodology.md](04_benchmark_methodology.md) | Benchmark 方法论与指标口径 |

## 求职文档

| 文件 | 内容 |
|---|---|
| [../RESUME_GUIDE.md](../RESUME_GUIDE.md) | **简历写法**：三个版本 + 5分钟口述稿 + 绝对不要写的 |
| [05_interview_qa.md](05_interview_qa.md) | 项目问答手册（针对这个项目的追问） |
| [qbank/](qbank/) | **定向题库**：43 题，CUDA/TensorRT/量化/引擎/Orin/多模态 |

## 阅读顺序建议

**上板前**：PROJECT_OVERVIEW → README → 01_setup

**调试中**：01_setup（按报错搜）→ 02/03（对应环节）

**跑出数据后**：04_benchmark（确认口径）→ 填 results/RESULTS_TEMPLATE.md

**准备面试**：RESUME_GUIDE → 05_interview_qa → qbank
