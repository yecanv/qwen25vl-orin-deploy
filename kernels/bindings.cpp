// PyTorch 扩展绑定
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>   // at::cuda::getCurrentCUDAStream 的声明在这里，torch/extension.h 不带
#include <cuda_runtime.h>

void launch_fused_match(const at::Half* tokens, int N, int D,
                        int* out_idx, float* out_val, cudaStream_t s);
void launch_merge(const at::Half* tokens, int N, int D,
                  const int* dst_of_a, float* accum, int* cnt, cudaStream_t s);
void launch_finalize(const at::Half* tokens, int N, int D,
                     const int* out_slot, const float* accum, const int* cnt,
                     at::Half* out, cudaStream_t s);

std::tuple<torch::Tensor, torch::Tensor> fused_match(torch::Tensor tokens) {
    TORCH_CHECK(tokens.is_cuda(), "tokens must be on CUDA");
    TORCH_CHECK(tokens.dtype() == torch::kHalf, "tokens must be fp16");
    TORCH_CHECK(tokens.dim() == 2, "tokens must be [N, D]");
    tokens = tokens.contiguous();
    int N = tokens.size(0), D = tokens.size(1);
    int Na = (N + 1) / 2;
    auto idx = torch::empty({Na}, tokens.options().dtype(torch::kInt32));
    auto val = torch::empty({Na}, tokens.options().dtype(torch::kFloat32));
    launch_fused_match(tokens.data_ptr<at::Half>(), N, D,
                       idx.data_ptr<int>(), val.data_ptr<float>(),
                       at::cuda::getCurrentCUDAStream());
    return {idx, val};
}

torch::Tensor merge_tokens(torch::Tensor tokens, torch::Tensor dst_of_a,
                           torch::Tensor out_slot, int n_kept) {
    tokens = tokens.contiguous();
    int N = tokens.size(0), D = tokens.size(1);
    int Nb = N / 2;
    auto accum = torch::zeros({Nb, D}, tokens.options().dtype(torch::kFloat32));
    auto cnt   = torch::zeros({Nb}, tokens.options().dtype(torch::kInt32));
    auto out   = torch::empty({n_kept, D}, tokens.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    launch_merge(tokens.data_ptr<at::Half>(), N, D, dst_of_a.data_ptr<int>(),
                 accum.data_ptr<float>(), cnt.data_ptr<int>(), stream);
    launch_finalize(tokens.data_ptr<at::Half>(), N, D, out_slot.data_ptr<int>(),
                    accum.data_ptr<float>(), cnt.data_ptr<int>(),
                    out.data_ptr<at::Half>(), stream);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_match", &fused_match, "融合 归一化+相似度+argmax");
    m.def("merge_tokens", &merge_tokens, "按合并计划输出压缩后的 token");
}
