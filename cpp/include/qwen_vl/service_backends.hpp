#pragma once

#include "qwen_vl/service.hpp"
#include "qwen_vl/llama_decoder.hpp"
#include <chrono>
#include <string>

namespace qwen_vl::service {
Backend make_synthetic_backend(std::chrono::milliseconds delay);

struct ModelBackendOptions {
    std::string vision_engine;
    std::string vision_contract;
    LlamaOptions llama;
};

Backend make_model_backend(const ModelBackendOptions& options);
}  // namespace qwen_vl::service
