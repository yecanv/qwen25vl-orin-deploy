#pragma once

#include "qwen_vl/types.hpp"

namespace qwen_vl {

struct PreprocessOptions {
    int target_width = 896;
    int target_height = 896;
};

// A single, already decoded, tightly packed RGB uint8 image. The explicit
// destination dimensions must be positive multiples of 28. This performs a
// fixed resize (it can change aspect ratio), not HF's dynamic smart_resize.
// The destination must match the shape used to build the TensorRT engine.
// Uses CLIP normalization and Qwen2.5-VL's 14/2/2 spatial/temporal/merge sizes.
// Invalid input throws std::invalid_argument; excessive sizes throw
// std::length_error or std::bad_alloc.
VisionInput preprocess(const Image& image, const PreprocessOptions& options = {});

}  // namespace qwen_vl
