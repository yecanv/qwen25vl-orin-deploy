#pragma once

#include <cstdint>
#include <vector>

namespace qwen_vl {

struct Image {
    int width = 0;
    int height = 0;
    std::vector<std::uint8_t> rgb;
};

// Patch grid BEFORE the 2x2 spatial merger. This runtime accepts still images (t=1).
struct Grid {
    int t = 1;
    int h = 0;
    int w = 0;
};

struct VisionInput {
    Grid grid;
    int patch_dim = 1176;
    std::vector<float> pixels;
};

struct VisionFeatures {
    int rows = 0;
    int hidden_size = 0;
    std::vector<float> values;
};

}  // namespace qwen_vl
