#pragma once

#include "qwen_vl/llama_decoder.hpp"
#include <string>
#include <vector>

namespace qwen_vl {
struct RuntimeOptions {
    std::string vision_engine;
    std::string vision_contract;
    std::string image;
    std::string prompt;
    std::string metrics;
    LlamaOptions llama;
    bool help = false;
};
// Paths are relative to the working directory, including paths in a config.
// Config is loaded first; explicitly supplied command-line values override it.
RuntimeOptions parse_options(const std::vector<std::string>& args);
std::string usage();
}  // namespace qwen_vl
