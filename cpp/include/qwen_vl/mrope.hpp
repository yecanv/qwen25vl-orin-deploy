#pragma once

#include "qwen_vl/types.hpp"

#include <cstdint>
#include <vector>

namespace qwen_vl {

struct ImageMrope {
    // llama.cpp's block-major layout: [t0..tN, h0..hN, w0..wN, 0..0].
    std::vector<std::int32_t> positions;
    int rows = 0;
    int merged_height = 0;
    int merged_width = 0;
    std::int32_t next_position = 0;
};

// grid describes patches BEFORE the 2x2 spatial merger. Single image only (t=1).
// Throws for invalid dimensions, non-divisible grids, or position/count overflow.
ImageMrope make_image_mrope(const Grid& grid, std::int32_t start_position,
                           int spatial_merge_size = 2);

}  // namespace qwen_vl
