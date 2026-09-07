#pragma once

#include "qwen_vl/types.hpp"
#include <string>

namespace qwen_vl {

struct EngineContract {
    Grid grid;
    int patch_dim = 1176;
    int hidden_size = 0;
    int target_width = 0;
    int target_height = 0;
};

Image read_image(const std::string& path);
EngineContract read_engine_contract(const std::string& path);
void write_pixels(const std::string& path, const VisionInput& input);

}  // namespace qwen_vl
