#include "qwen_vl/mrope.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace qwen_vl {

ImageMrope make_image_mrope(const Grid& grid, std::int32_t start_position,
                           int spatial_merge_size) {
    if (grid.t != 1) {
        throw std::invalid_argument("M-RoPE supports one still image only (grid.t must be 1)");
    }
    if (spatial_merge_size <= 0 || grid.h <= 0 || grid.w <= 0 || start_position < 0) {
        throw std::invalid_argument("M-RoPE grid/merge dimensions must be positive and start nonnegative");
    }
    if (grid.h % spatial_merge_size != 0 || grid.w % spatial_merge_size != 0) {
        throw std::invalid_argument("M-RoPE patch grid must be divisible by spatial_merge_size");
    }

    ImageMrope result;
    result.merged_height = grid.h / spatial_merge_size;
    result.merged_width = grid.w / spatial_merge_size;
    const auto count = static_cast<std::int64_t>(result.merged_height) * result.merged_width;
    const auto next = static_cast<std::int64_t>(start_position) +
                      std::max(result.merged_height, result.merged_width);
    if (count > std::numeric_limits<std::int32_t>::max() ||
        next > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("M-RoPE token count or position exceeds int32 capacity");
    }
    if (static_cast<std::uint64_t>(count) > result.positions.max_size() / 4U) {
        throw std::overflow_error("M-RoPE position buffer exceeds addressable memory");
    }
    result.rows = static_cast<int>(count);
    result.next_position = static_cast<std::int32_t>(next);
    const auto n = static_cast<std::size_t>(result.rows);
    result.positions.resize(4U * n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        result.positions[i] = start_position;
        result.positions[n + i] = start_position + static_cast<std::int32_t>(i / result.merged_width);
        result.positions[2U * n + i] = start_position + static_cast<std::int32_t>(i % result.merged_width);
    }
    return result;
}

}  // namespace qwen_vl
