// =============================================================================
// token_merge_ref.cpp —— CUDA kernel 的 CPU 镜像实现
//
// 目的：在没有 GPU 的环境下验证算法逻辑。
//
// 这不是"另一种写法"，而是**逐行对应** token_merge.cu 的：
//   - 同样的 BLOCK_M / BLOCK_N / BLOCK_K 分块
//   - 同样的线程→(m, n) 映射（m = tid/8, n = tid%8 + 8k）
//   - 同样的归约顺序（模拟 __shfl_down_sync width=8）
//   - 同样的平局处理规则
//
// 能验证：分块边界、跨步取列、归约顺序、平局确定性、合并逻辑
// 不能验证：CUDA 语法、shared memory bank conflict、实际性能
//
// 编译：g++ -O2 -std=c++17 token_merge_ref.cpp -o ref && ./ref
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>
#include <cfloat>

#define BLOCK_M 32
#define BLOCK_N 32
#define BLOCK_K 32
#define THREADS 256
#define ACC_PER_THREAD (BLOCK_M * BLOCK_N / THREADS)
#define LANES_PER_ROW  (THREADS / BLOCK_M)

// -----------------------------------------------------------------------------
// 参考实现 A：朴素三段式（等价于 bench_kernel.py 里 PyTorch 模拟的朴素路径）
// -----------------------------------------------------------------------------
void naive_match(const std::vector<float>& tok, int N, int D,
                 std::vector<int>& idx, std::vector<float>& val)
{
    int Na = (N + 1) / 2, Nb = N / 2;
    idx.assign(Na, -1);
    val.assign(Na, -FLT_MAX);

    // ① 归一化
    std::vector<float> na(Na), nb(Nb);
    for (int m = 0; m < Na; ++m) {
        double s = 0;
        for (int d = 0; d < D; ++d) { double v = tok[(size_t)(2*m)*D + d]; s += v*v; }
        na[m] = (float)std::sqrt(s) + 1e-6f;
    }
    for (int n = 0; n < Nb; ++n) {
        double s = 0;
        for (int d = 0; d < D; ++d) { double v = tok[(size_t)(2*n+1)*D + d]; s += v*v; }
        nb[n] = (float)std::sqrt(s) + 1e-6f;
    }

    // ② 完整相似度矩阵（这就是融合版要省掉的中间产物）
    std::vector<float> S((size_t)Na * Nb);
    for (int m = 0; m < Na; ++m)
        for (int n = 0; n < Nb; ++n) {
            float acc = 0.f;
            for (int d = 0; d < D; ++d)
                acc += tok[(size_t)(2*m)*D + d] * tok[(size_t)(2*n+1)*D + d];
            S[(size_t)m*Nb + n] = acc / (na[m] * nb[n]);
        }

    // ③ 逐行 argmax
    for (int m = 0; m < Na; ++m) {
        float bv = -FLT_MAX; int bi = -1;
        for (int n = 0; n < Nb; ++n) {
            float v = S[(size_t)m*Nb + n];
            if (v > bv || (v == bv && bi >= 0 && n < bi)) { bv = v; bi = n; }
        }
        val[m] = bv; idx[m] = bi;
    }
}

// -----------------------------------------------------------------------------
// 参考实现 B：融合版的 CPU 镜像
// 严格模拟 CUDA kernel 的 block / thread / warp-shuffle 结构
// -----------------------------------------------------------------------------
void fused_match_cpu(const std::vector<float>& tok, int N, int D,
                     std::vector<int>& idx, std::vector<float>& val)
{
    int Na = (N + 1) / 2, Nb = N / 2;
    idx.assign(Na, -1);
    val.assign(Na, -FLT_MAX);

    int n_blocks = (Na + BLOCK_M - 1) / BLOCK_M;

    for (int blk = 0; blk < n_blocks; ++blk) {          // ← gridDim.x
        int m_base = blk * BLOCK_M;

        float As[BLOCK_M][BLOCK_K];                     // ← __shared__
        float Bs[BLOCK_N][BLOCK_K];
        float a_norm[BLOCK_M], b_norm[BLOCK_N];

        // ---- A 行范数（模拟 8-lane shuffle 归约）----
        for (int m_local = 0; m_local < BLOCK_M; ++m_local) {
            int mg = m_base + m_local;
            bool valid = mg < Na;
            // 每个 lane 各自累加自己负责的 d
            float lane[LANES_PER_ROW] = {0};
            for (int L = 0; L < LANES_PER_ROW; ++L)
                for (int d = L; d < D; d += LANES_PER_ROW) {
                    float v = valid ? tok[(size_t)(2*mg)*D + d] : 0.f;
                    lane[L] += v * v;
                }
            // 模拟 __shfl_down_sync(width=8)：offset 4,2,1
            for (int off = LANES_PER_ROW/2; off > 0; off >>= 1)
                for (int L = 0; L < off; ++L) lane[L] += lane[L + off];
            a_norm[m_local] = std::sqrt(lane[0]) + 1e-6f;
        }

        // 每线程的 running max
        float best_v[THREADS];
        int   best_i[THREADS];
        for (int t = 0; t < THREADS; ++t) { best_v[t] = -FLT_MAX; best_i[t] = -1; }

        for (int n_base = 0; n_base < Nb; n_base += BLOCK_N) {

            // ---- B tile 范数 ----
            for (int n_local = 0; n_local < BLOCK_N; ++n_local) {
                int ng = n_base + n_local;
                bool valid = ng < Nb;
                float lane[LANES_PER_ROW] = {0};
                for (int L = 0; L < LANES_PER_ROW; ++L)
                    for (int d = L; d < D; d += LANES_PER_ROW) {
                        float v = valid ? tok[(size_t)(2*ng+1)*D + d] : 0.f;
                        lane[L] += v * v;
                    }
                for (int off = LANES_PER_ROW/2; off > 0; off >>= 1)
                    for (int L = 0; L < off; ++L) lane[L] += lane[L + off];
                b_norm[n_local] = std::sqrt(lane[0]) + 1e-6f;
            }

            // ---- 累加器 ----
            std::vector<std::vector<float>> acc(
                THREADS, std::vector<float>(ACC_PER_THREAD, 0.f));

            // ---- K 分块 ----
            for (int k_base = 0; k_base < D; k_base += BLOCK_K) {
                // 载入 tile（含边界置零）
                for (int r = 0; r < BLOCK_M; ++r)
                    for (int c = 0; c < BLOCK_K; ++c) {
                        int mg = m_base + r, kk = k_base + c;
                        As[r][c] = (mg < Na && kk < D)
                                 ? tok[(size_t)(2*mg)*D + kk] : 0.f;
                    }
                for (int r = 0; r < BLOCK_N; ++r)
                    for (int c = 0; c < BLOCK_K; ++c) {
                        int ng = n_base + r, kk = k_base + c;
                        Bs[r][c] = (ng < Nb && kk < D)
                                 ? tok[(size_t)(2*ng+1)*D + kk] : 0.f;
                    }

                // 外积累加（严格按线程映射）
                for (int t = 0; t < THREADS; ++t) {
                    int m_local = t / LANES_PER_ROW;
                    int lane_in_row = t % LANES_PER_ROW;
                    for (int i = 0; i < ACC_PER_THREAD; ++i) {
                        int n_local = lane_in_row + i * LANES_PER_ROW;
                        float s = 0.f;
                        for (int k = 0; k < BLOCK_K; ++k)
                            s += As[m_local][k] * Bs[n_local][k];
                        acc[t][i] += s;
                    }
                }
            }

            // ---- 归一化 + 更新 running max ----
            for (int t = 0; t < THREADS; ++t) {
                int m_local = t / LANES_PER_ROW;
                int lane_in_row = t % LANES_PER_ROW;
                for (int i = 0; i < ACC_PER_THREAD; ++i) {
                    int n_local = lane_in_row + i * LANES_PER_ROW;
                    int n_global = n_base + n_local;
                    if (n_global >= Nb) continue;
                    float cs = acc[t][i] / (a_norm[m_local] * b_norm[n_local]);
                    if (cs > best_v[t]) { best_v[t] = cs; best_i[t] = n_global; }
                }
            }
        }

        // ---- 行内 8-lane 归约（模拟 __shfl_down_sync）----
        for (int m_local = 0; m_local < BLOCK_M; ++m_local) {
            int t0 = m_local * LANES_PER_ROW;
            for (int off = LANES_PER_ROW/2; off > 0; off >>= 1)
                for (int L = 0; L < off; ++L) {
                    int a = t0 + L, b = t0 + L + off;
                    if (best_v[b] > best_v[a] ||
                        (best_v[b] == best_v[a] && best_i[b] >= 0
                         && best_i[b] < best_i[a])) {
                        best_v[a] = best_v[b];
                        best_i[a] = best_i[b];
                    }
                }
            int mg = m_base + m_local;
            if (mg < Na) { idx[mg] = best_i[t0]; val[mg] = best_v[t0]; }
        }
    }
}

// -----------------------------------------------------------------------------
// 合并阶段（对应 merge_scatter + finalize）
// -----------------------------------------------------------------------------
struct MergePlan {
    std::vector<int> dst_of_a;    // [Na]  -1 = 不合并
    std::vector<int> out_slot;    // [N]   -1 = 丢弃
    int n_kept;
};

MergePlan make_plan(const std::vector<int>& idx, const std::vector<float>& val,
                    int N, int r)
{
    int Na = (N + 1) / 2, Nb = N / 2;
    MergePlan p;
    p.dst_of_a.assign(Na, -1);
    p.out_slot.assign(N, -1);

    // 按相似度降序取 top-r
    std::vector<int> order(Na);
    for (int i = 0; i < Na; ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(),
        [&](int a, int b){ return val[a] > val[b]; });

    int r_eff = std::min(r, Na);
    for (int k = 0; k < r_eff; ++k) p.dst_of_a[order[k]] = idx[order[k]];

    int slot = 0;
    for (int src = 0; src < N; ++src) {
        if (src % 2 == 0) {                       // A token
            int m = src / 2;
            if (m < Na && p.dst_of_a[m] >= 0) continue;   // 被合并掉
        }
        p.out_slot[src] = slot++;
    }
    p.n_kept = slot;
    return p;
}

void apply_merge(const std::vector<float>& tok, int N, int D,
                 const MergePlan& p, std::vector<float>& out)
{
    int Na = (N + 1) / 2, Nb = N / 2;
    std::vector<float> accum((size_t)Nb * D, 0.f);
    std::vector<int> cnt(Nb, 0);

    for (int m = 0; m < Na; ++m) {
        int dst = p.dst_of_a[m];
        if (dst < 0) continue;
        for (int d = 0; d < D; ++d)
            accum[(size_t)dst*D + d] += tok[(size_t)(2*m)*D + d];
        cnt[dst]++;
    }

    out.assign((size_t)p.n_kept * D, 0.f);
    for (int src = 0; src < N; ++src) {
        int slot = p.out_slot[src];
        if (slot < 0) continue;
        bool is_b = (src & 1);
        int b_idx = src >> 1;
        int c = (is_b && b_idx < Nb) ? cnt[b_idx] : 0;
        for (int d = 0; d < D; ++d) {
            float v;
            if (is_b && c > 0)
                v = (tok[(size_t)src*D + d] + accum[(size_t)b_idx*D + d]) / (c + 1);
            else
                v = tok[(size_t)src*D + d];
            out[(size_t)slot*D + d] = v;
        }
    }
}

// -----------------------------------------------------------------------------
// 测试
// -----------------------------------------------------------------------------
int run_case(int N, int D, unsigned seed, bool verbose)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> tok((size_t)N * D);
    for (auto& v : tok) v = nd(rng);

    // 制造一些高相似度对，检验匹配是否找得到
    int n_planted = std::min(4, N / 4);
    std::vector<std::pair<int,int>> planted;
    for (int i = 0; i < n_planted; ++i) {
        int m = (i * 7) % ((N+1)/2);
        int n = (i * 5 + 3) % (N/2);
        for (int d = 0; d < D; ++d)
            tok[(size_t)(2*n+1)*D + d] = tok[(size_t)(2*m)*D + d] * 1.3f
                                       + nd(rng) * 0.01f;
        planted.push_back({m, n});
    }

    std::vector<int> idx_n, idx_f;
    std::vector<float> val_n, val_f;
    naive_match(tok, N, D, idx_n, val_n);
    fused_match_cpu(tok, N, D, idx_f, val_f);

    int Na = (N + 1) / 2;
    int idx_mismatch = 0;
    double max_val_diff = 0;
    for (int m = 0; m < Na; ++m) {
        if (idx_n[m] != idx_f[m]) idx_mismatch++;
        max_val_diff = std::max(max_val_diff,
                                (double)std::fabs(val_n[m] - val_f[m]));
    }

    // 植入的高相似对是否被找到
    int planted_found = 0;
    for (auto& pr : planted)
        if (idx_f[pr.first] == pr.second) planted_found++;

    // 合并链路
    int r = Na / 2;
    MergePlan plan = make_plan(idx_f, val_f, N, r);
    std::vector<float> out;
    apply_merge(tok, N, D, plan, out);
    int expect_kept = N - std::min(r, Na);

    bool ok = (idx_mismatch == 0) && (max_val_diff < 2e-3)
              && (plan.n_kept == expect_kept)
              && (planted_found == (int)planted.size());

    printf("N=%5d D=%5d | argmax不一致 %3d | 相似度最大误差 %.2e | "
           "植入对命中 %d/%d | 压缩 %d->%d (期望 %d) | %s\n",
           N, D, idx_mismatch, max_val_diff,
           planted_found, (int)planted.size(),
           N, plan.n_kept, expect_kept, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int main()
{
    printf("=== 融合算子 CPU 镜像 vs 朴素三段式 对拍 ===\n");
    printf("（验证分块边界、跨步取列、8-lane 归约顺序、平局规则、合并链路）\n\n");
    int fails = 0;
    // 覆盖：整除 / 不整除 BLOCK_M / 不整除 BLOCK_K / 奇数 N / 大 D
    fails += run_case(64,   64,  1, true);
    fails += run_case(128,  128, 2, true);
    fails += run_case(100,  96,  3, true);   // N 不整除 32
    fails += run_case(129,  64,  4, true);   // N 为奇数
    fails += run_case(256,  100, 5, true);   // D 不整除 32
    fails += run_case(37,   33,  6, true);   // 极小且都不整除
    fails += run_case(512,  256, 7, true);
    fails += run_case(1024, 128, 8, true);   // 接近真实视觉 token 规模

    printf("\n%s\n", fails == 0 ? "全部通过" : "存在失败用例");
    return fails;
}
