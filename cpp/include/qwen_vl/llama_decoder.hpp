#pragma once

#include "qwen_vl/types.hpp"

#include <functional>
#include <memory>
#include <string>

namespace qwen_vl {

struct LlamaOptions {
    std::string model_path;
    int n_ctx = 4096;
    int n_batch = 2048;
    int n_gpu_layers = 99;
    int max_new_tokens = 256;
    int n_threads = 0;  // 0: use llama.cpp's default.
};

struct GenerationTiming {
    double model_load_ms = 0.0;
    double context_init_ms = 0.0;
    double tokenize_ms = 0.0;
    double prefill_ms = 0.0;
    // generate() entry to first non-EOG token sampling; excludes model load and vision.
    // -1 means no non-EOG token was generated (including prefill-only requests).
    double first_token_ms = -1.0;
    double generation_ms = 0.0;
    // Time after the first non-EOG token; includes callback time and terminal sampling.
    double decode_ms = 0.0;
};

struct GenerationResult {
    std::string text;
    int generated_tokens = 0;
    int decoded_tokens = 0;  // Actual one-token llama_decode calls after prefill.
    bool stopped_on_eog = false;
    GenerationTiming timing;
};

struct GenerationControl {
    int max_new_tokens = -1;
    std::function<bool()> stop_requested;
};

// Loads the GGUF once; each generate call creates an isolated context and uses greedy
// decoding. This class supports single-image requests; calls must not overlap.
class LlamaDecoder {
public:
    explicit LlamaDecoder(LlamaOptions options);
    ~LlamaDecoder();
    LlamaDecoder(LlamaDecoder&&) noexcept;
    LlamaDecoder& operator=(LlamaDecoder&&) noexcept;
    LlamaDecoder(const LlamaDecoder&) = delete;
    LlamaDecoder& operator=(const LlamaDecoder&) = delete;

    GenerationResult generate(
        const VisionFeatures& features, const Grid& grid, const std::string& question,
        const std::function<void(const std::string&)>& on_piece = {},
        const GenerationControl& control = {});

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qwen_vl
