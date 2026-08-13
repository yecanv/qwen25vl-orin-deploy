// =============================================================================
// token_merge_launcher.cu —— kernel 启动包装
// setup.py 编译这个文件，它 include token_merge.cu 里的 kernel 定义
// =============================================================================
#include "token_merge.cu"   // BLOCK_M / THREADS 等宏随之引入，不在这里重复定义
#include <ATen/ATen.h>

void launch_fused_match(const at::Half* tokens, int N, int D,
                        int* out_idx, float* out_val, cudaStream_t s)
{
    int Na = (N + 1) / 2;
    int grid = (Na + BLOCK_M - 1) / BLOCK_M;
    fused_match_kernel<<<grid, THREADS, 0, s>>>(
        reinterpret_cast<const __half*>(tokens), N, D, out_idx, out_val);
}

void launch_merge(const at::Half* tokens, int N, int D,
                  const int* dst_of_a, float* accum, int* cnt, cudaStream_t s)
{
    int Na = (N + 1) / 2;
    merge_scatter_kernel<<<Na, THREADS, 0, s>>>(
        reinterpret_cast<const __half*>(tokens), N, D, dst_of_a, accum, cnt);
}

void launch_finalize(const at::Half* tokens, int N, int D,
                     const int* out_slot, const float* accum, const int* cnt,
                     at::Half* out, cudaStream_t s)
{
    finalize_kernel<<<N, THREADS, 0, s>>>(
        reinterpret_cast<const __half*>(tokens), N, D, out_slot, accum, cnt,
        reinterpret_cast<__half*>(out));
}
