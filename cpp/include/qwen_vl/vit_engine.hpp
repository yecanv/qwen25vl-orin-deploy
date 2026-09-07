#pragma once

#include "qwen_vl/types.hpp"

#include <memory>
#include <string>

namespace qwen_vl {

// Owns a TensorRT 10.3 vision engine and reusable CUDA buffers.
// Only the fixed-input export_vit_onnx_static.py contract is supported.
// Runtime-reported output dimensions are checked through IOutputAllocator.
// Calls to infer() on one instance must be serialized.
class VitEngine {
public:
    // The default hidden size is the Qwen2.5-VL-3B shapes.json contract.
    explicit VitEngine(const std::string& engine_path, int expected_hidden_size = 2048);
    ~VitEngine();

    VitEngine(const VitEngine&) = delete;
    VitEngine& operator=(const VitEngine&) = delete;
    VitEngine(VitEngine&&) = delete;
    VitEngine& operator=(VitEngine&&) = delete;

    int input_patches() const;
    int output_rows() const;
    int hidden_size() const;

    // The caller must use the exact grid baked into the engine at export.
    // Tensor dimensions identify the patch count, but cannot distinguish
    // different grids with the same area (for example, 64x64 and 32x128).
    VisionFeatures infer(const VisionInput& input);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace qwen_vl
