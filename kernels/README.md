# 视觉 token 压缩融合算子

## 这个模块解决什么问题

Qwen2.5-VL 在高分辨率输入下，视觉 token 占 prompt 的绝大部分。
在 Orin 上这直接决定两件事：

- **TTFT** —— prefill 的 attention 是 O(n²)
- **可用并发** —— KV Cache 占用与 token 数成正比

所以压缩视觉 token 是这个项目里收益最大的优化方向，
而不是一个为了"简历上要有 CUDA"硬凑的练习题。

## 文件

```
token_merge.cu             融合 kernel（归一化 + 相似度 + argmax）+ 合并 kernel
                           另含朴素三段式实现，作为 benchmark 对照组
ref/token_merge_ref.cpp    CPU 镜像实现，逐行对应 CUDA 的分块与归约顺序
                           用于无 GPU 环境下验证算法正确性
bindings.cpp               PyTorch 扩展绑定
setup.py                   编译脚本（Orin 用 CUDA_ARCH=87）
bench_kernel.py            kernel 级 + 端到端 benchmark
```

## 先验证再上板

CPU 镜像可以在任何机器上跑，先确认算法对：

```bash
cd ref && g++ -O2 -std=c++17 token_merge_ref.cpp -o ref && ./ref
```

已验证 8 组用例全部通过，覆盖：
N 不整除 BLOCK_M、N 为奇数、D 不整除 BLOCK_K、极小尺寸、接近真实规模。
融合版与朴素三段式的 argmax 结果**完全一致**，相似度最大误差 < 6e-7。

## 上板

```bash
CUDA_ARCH=87 python setup.py install
python bench_kernel.py --n-tokens 1024 --dim 2048
```

## 融合收益的诚实说明

**融合省的是访存和 kernel launch 开销，GEMM 的算力开销一分没省。**

省下的部分：
- 相似度矩阵 S 的一写一读（S 是纯中间产物，我们只要每行的 max）
- 归一化结果的中间落地
- 2 次 kernel launch（Orin 的 CPU 弱，launch 开销比桌面卡显著）

离线估算（Orin Nano，带宽按标称 78% 计）：

| 视觉 token | S 矩阵 | 省下访存 | 占 kernel 总耗时 |
|---|---|---|---|
| 256 | 0.1 MB | 3.3 MB | ~87% |
| 1024 | 1.0 MB | 14.7 MB | ~52% |
| 2048 | 4.2 MB | 33.6 MB | ~36% |
| 4096 | 16.8 MB | 83.9 MB | ~24% |

N 越小 launch 开销占比越高，N 越大 S 矩阵访存占比越高。
**实际数字必须在你自己板子上用 bench_kernel.py 测，上表只是数量级参考。**

**端到端的大头不在这个 kernel**，而在于 token 数减少后 LLM prefill 的
attention 计算量按平方降：

| 压缩率 | 剩余 token | attention 计算量 | KV Cache |
|---|---|---|---|
| 0% | 100% | 100% | 100% |
| 25% | 75% | 56% | 75% |
| 50% | 50% | 25% | 50% |

面试讲这个项目时，**要把这两层收益分开说**。
只说"我写了个 kernel 快了多少"是浅的；
说清楚"kernel 省了访存和 launch、但真正的收益来自 O(n²) 的降维"
才显出你知道瓶颈在哪。
