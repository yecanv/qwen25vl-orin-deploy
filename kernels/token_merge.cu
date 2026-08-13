// =============================================================================
// 视觉 token 压缩：融合算子
// token_merge.cu
//
// 背景
// ----
// Qwen2.5-VL 在高分辨率输入下，视觉 token 占 prompt 的绝大部分
// （1024x1024 的图 → 约 1300 个视觉 token，而文本 prompt 通常只有几十个）。
// 这直接主导两件事：
//   1. prefill 的计算量（attention 是 O(n²)）→ TTFT
//   2. KV Cache 显存占用 → 可用并发数
// 在 Orin 这种显存和算力都紧张的平台上，压缩视觉 token 是收益最大的优化方向。
//
// 算法：双向图软匹配（ToMe 思路，针对 Qwen2-VL 的空间结构调整）
// ----------------------------------------------------------------
//   1. 把 N 个 token 按下标奇偶拆成两组 A（偶）、B（奇）
//      —— 拆奇偶而不是拆前后半，是因为 Qwen2-VL 的 token 按 raster order
//         排列，奇偶拆分能保证 A 中每个 token 的空间邻居大概率落在 B 里
//   2. 对每个 a ∈ A，找 cos 相似度最高的 b ∈ B，记录 (best_score, best_idx)
//   3. 取 score 最高的 r 个 a，合并进各自匹配的 b（简单平均：(b + Σa)/(cnt+1)）
//   4. 剩余 N - r 个 token 送入 LLM
//
// 为什么要融合
// ------------
// 朴素实现要 3 次 kernel launch：
//   ① L2 归一化           读 N·D 写 N·D
//   ② 相似度矩阵 S = Â·B̂ᵀ  写 (N/2)² 个 float   ← 主要开销
//   ③ 逐行 argmax          读 (N/2)²
//
// S 矩阵本身完全是中间产物，我们只要每行的 max 和 argmax。
// 融合后 S 从不落地：在寄存器里累加完一个 tile 就立刻并入 running max。
//
// 收益来源（诚实版，别夸大）：
//   - 省掉 S 的一写一读：2·(N/2)²·4 bytes。N=2048 时约 8 MB
//   - 省掉 2 次 kernel launch。Orin 的 CPU 弱，launch 开销比桌面卡显著
//   - 归一化融进 K-loop，省一趟 N·D 的读写
//   GEMM 本身的算力开销不变 —— 这不是一个把计算量降下来的优化，
//   而是一个把访存和 launch 开销降下来的优化。实测收益见 bench_kernel.py。
//
// 分块设计
// --------
//   BLOCK_M = 32   每个 block 负责 32 个 A 行
//   BLOCK_N = 32   每次载入 32 个 B 行
//   BLOCK_K = 32   K 方向每次处理 32 维
//   blockDim = 256 (8 warps)，每线程持有 4 个累加器 (32×32/256 = 4)
//
//   线程 t 的映射：  m = t / 8            （32 行，每行 8 个线程）
//                   n = (t % 8) + 8·k    （k = 0..3，跨步取列）
//   跨步取列是为了让同一行的 8 个线程落在同一个 warp 的连续 lane 上，
//   这样行内归约可以直接用 __shfl_down_sync(width=8)，不走 shared memory。
//
//   共享内存：As[32][33] + Bs[32][33]，pad 到 33 避免 bank conflict
//             = 2 × 32 × 33 × 4 B = 8.25 KB / block
// =============================================================================

#pragma once
// 注意：本文件被 token_merge_launcher.cu 以 #include 方式引入（kernel 定义随包装函数一起编译），
// 所以按 header 惯例把 pragma once 放在文件最前。

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>

#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32
#define THREADS 256
#define ACC_PER_THREAD (BLOCK_M * BLOCK_N / THREADS)   // = 4
#define LANES_PER_ROW  (THREADS / BLOCK_M)             // = 8
#define SMEM_PAD 33                                    // 32 + 1，避开 bank conflict

// -----------------------------------------------------------------------------
// Kernel 1：融合 归一化 + 相似度 + 逐行 argmax
//
// tokens : [N, D]  half，视觉 token（merger 输出）
// out_idx: [Na]    int，每个 A token 匹配到的 B token 下标（B 内局部下标）
// out_val: [Na]    float，对应的 cos 相似度
// -----------------------------------------------------------------------------
__global__ void fused_match_kernel(
    const __half* __restrict__ tokens,
    int N, int D,
    int* __restrict__ out_idx,
    float* __restrict__ out_val)
{
    __shared__ float As[BLOCK_M][SMEM_PAD];
    __shared__ float Bs[BLOCK_N][SMEM_PAD];
    __shared__ float a_norm[BLOCK_M];
    __shared__ float b_norm[BLOCK_N];

    const int tid  = threadIdx.x;
    const int m_local = tid / LANES_PER_ROW;        // 0..31
    const int lane_in_row = tid % LANES_PER_ROW;    // 0..7

    const int Na = (N + 1) / 2;   // A 取偶数下标
    const int Nb = N / 2;         // B 取奇数下标

    const int m_base = blockIdx.x * BLOCK_M;
    const int m_global = m_base + m_local;
    const bool m_valid = (m_global < Na);

    // A 行在原 tokens 里的下标 = 2 * m_global
    const int a_row = m_valid ? (2 * m_global) : 0;

    // ---- 先算 A 行的 L2 范数（融合进来，省一趟独立 kernel）----
    {
        float s = 0.f;
        for (int d = lane_in_row; d < D; d += LANES_PER_ROW) {
            float v = m_valid ? __half2float(tokens[(long)a_row * D + d]) : 0.f;
            s += v * v;
        }
        // 行内 8 lane 归约
        #pragma unroll
        for (int off = LANES_PER_ROW / 2; off > 0; off >>= 1)
            s += __shfl_down_sync(0xffffffffu, s, off, LANES_PER_ROW);
        if (lane_in_row == 0) a_norm[m_local] = sqrtf(s) + 1e-6f;
    }
    __syncthreads();

    // ---- running max（每线程先各自维护，最后行内归约）----
    float best_v = -FLT_MAX;
    int   best_i = -1;

    for (int n_base = 0; n_base < Nb; n_base += BLOCK_N) {

        // ---- B tile 的 L2 范数 ----
        {
            int n_local = tid / LANES_PER_ROW;
            int n_global = n_base + n_local;
            bool n_valid = (n_global < Nb);
            int b_row = n_valid ? (2 * n_global + 1) : 0;
            float s = 0.f;
            for (int d = lane_in_row; d < D; d += LANES_PER_ROW) {
                float v = n_valid ? __half2float(tokens[(long)b_row * D + d]) : 0.f;
                s += v * v;
            }
            #pragma unroll
            for (int off = LANES_PER_ROW / 2; off > 0; off >>= 1)
                s += __shfl_down_sync(0xffffffffu, s, off, LANES_PER_ROW);
            if (lane_in_row == 0) b_norm[n_local] = sqrtf(s) + 1e-6f;
        }
        __syncthreads();

        // ---- 累加器清零 ----
        float acc[ACC_PER_THREAD];
        #pragma unroll
        for (int i = 0; i < ACC_PER_THREAD; ++i) acc[i] = 0.f;

        // ---- K 方向分块 ----
        for (int k_base = 0; k_base < D; k_base += BLOCK_K) {

            // 载入 A tile：256 线程搬 32×32
            {
                int r = tid / BLOCK_K;          // 0..7
                int c = tid % BLOCK_K;          // 0..31
                #pragma unroll
                for (int rr = r; rr < BLOCK_M; rr += THREADS / BLOCK_K) {
                    int mg = m_base + rr;
                    int kk = k_base + c;
                    As[rr][c] = (mg < Na && kk < D)
                        ? __half2float(tokens[(long)(2 * mg) * D + kk]) : 0.f;
                }
            }
            // 载入 B tile
            {
                int r = tid / BLOCK_K;
                int c = tid % BLOCK_K;
                #pragma unroll
                for (int rr = r; rr < BLOCK_N; rr += THREADS / BLOCK_K) {
                    int ng = n_base + rr;
                    int kk = k_base + c;
                    Bs[rr][c] = (ng < Nb && kk < D)
                        ? __half2float(tokens[(long)(2 * ng + 1) * D + kk]) : 0.f;
                }
            }
            __syncthreads();

            // ---- 外积累加 ----
            #pragma unroll
            for (int i = 0; i < ACC_PER_THREAD; ++i) {
                int n_local = lane_in_row + i * LANES_PER_ROW;   // 跨步取列
                float s = 0.f;
                #pragma unroll
                for (int k = 0; k < BLOCK_K; ++k)
                    s += As[m_local][k] * Bs[n_local][k];
                acc[i] += s;
            }
            __syncthreads();
        }

        // ---- 归一化并更新 running max（S 从不写回全局内存）----
        #pragma unroll
        for (int i = 0; i < ACC_PER_THREAD; ++i) {
            int n_local  = lane_in_row + i * LANES_PER_ROW;
            int n_global = n_base + n_local;
            if (n_global >= Nb) continue;
            float cs = acc[i] / (a_norm[m_local] * b_norm[n_local]);
            if (cs > best_v) { best_v = cs; best_i = n_global; }
        }
        __syncthreads();
    }

    // ---- 行内 8 lane 归约出最终 argmax ----
    #pragma unroll
    for (int off = LANES_PER_ROW / 2; off > 0; off >>= 1) {
        float ov = __shfl_down_sync(0xffffffffu, best_v, off, LANES_PER_ROW);
        int   oi = __shfl_down_sync(0xffffffffu, best_i, off, LANES_PER_ROW);
        // 平局时取下标小的，保证结果确定性（可复现，便于和 CPU 参考对拍）
        if (ov > best_v || (ov == best_v && oi >= 0 && oi < best_i)) {
            best_v = ov; best_i = oi;
        }
    }

    if (lane_in_row == 0 && m_valid) {
        out_idx[m_global] = best_i;
        out_val[m_global] = best_v;
    }
}


// -----------------------------------------------------------------------------
// Kernel 2：按合并计划做 gather-merge
//
// keep_mask : [N]  uint8，1 = 保留，0 = 被合并掉
// dst_of_a  : [Na] int，A[m] 要并进的 B 下标；-1 表示不合并
// merge_cnt : [Nb] int，每个 B token 吸收了多少个 A（含自己 = cnt+1）
//
// 用 atomicAdd 累加。注意标量 __half 的 atomicAdd 需要 SM70+（SM60 只提供
// __half2 向量版），Orin 是 SM87，没问题。
// 但为了数值稳定，这里在 float 缓冲区上累加，最后再转回 half。
// -----------------------------------------------------------------------------
__global__ void merge_scatter_kernel(
    const __half* __restrict__ tokens,
    int N, int D,
    const int* __restrict__ dst_of_a,
    float* __restrict__ accum,          // [Nb, D] float 缓冲
    int* __restrict__ merge_cnt)        // [Nb]
{
    const int Na = (N + 1) / 2;
    int m = blockIdx.x;
    if (m >= Na) return;

    int dst = dst_of_a[m];
    if (dst < 0) return;                // 该 A token 保留，不合并

    int a_row = 2 * m;
    for (int d = threadIdx.x; d < D; d += blockDim.x)
        atomicAdd(&accum[(long)dst * D + d],
                  __half2float(tokens[(long)a_row * D + d]));

    if (threadIdx.x == 0) atomicAdd(&merge_cnt[dst], 1);
}


// -----------------------------------------------------------------------------
// Kernel 3：写出压缩后的 token 序列
// -----------------------------------------------------------------------------
__global__ void finalize_kernel(
    const __half* __restrict__ tokens,
    int N, int D,
    const int* __restrict__ out_slot,   // [N] 每个原 token 的目标位置，-1 = 丢弃
    const float* __restrict__ accum,    // [Nb, D]
    const int* __restrict__ merge_cnt,  // [Nb]
    __half* __restrict__ out)           // [N_kept, D]
{
    int src = blockIdx.x;
    if (src >= N) return;
    int slot = out_slot[src];
    if (slot < 0) return;

    bool is_b = (src & 1);
    int  b_idx = src >> 1;
    int  cnt = is_b ? merge_cnt[b_idx] : 0;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float v;
        if (is_b && cnt > 0) {
            // 自己 + 被吸收的 A，取平均
            v = (__half2float(tokens[(long)src * D + d])
                 + accum[(long)b_idx * D + d]) / (float)(cnt + 1);
        } else {
            v = __half2float(tokens[(long)src * D + d]);
        }
        out[(long)slot * D + d] = __float2half(v);
    }
}

// 朴素三段式对照组不在 GPU 侧单独实现：
// benchmark 对比用 bench_kernel.py 里的 PyTorch 算子组合（独立归一化 →
// 物化完整 S 矩阵 → 逐行 argmax），数值对拍用 ref/token_merge_ref.cpp 的 CPU 实现。
