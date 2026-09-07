#include "qwen_vl/vit_engine.hpp"

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen_vl {
namespace {

constexpr const char* kInputName = "pixel_values";
constexpr const char* kOutputName = "vision_embeds";
constexpr int kPatchDim = 3 * 2 * 14 * 14;
constexpr int kMergeArea = 4;

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

std::size_t checked_product(std::size_t lhs, std::size_t rhs) {
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs) {
        throw std::overflow_error("Tensor element/byte count exceeds size_t");
    }
    return lhs * rhs;
}

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            // A TensorRT callback must never propagate a C++ exception.
            try {
                std::cerr << "[TensorRT] " << (message ? message : "(no message)") << '\n';
            } catch (...) {
            }
        }
    }
};

class CudaStream {
public:
    CudaStream() = default;
    ~CudaStream() {
        if (stream_) {
            (void)cudaStreamDestroy(stream_);
        }
    }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;
    void create() {
        check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
    }
    cudaStream_t get() const { return stream_; }

private:
    cudaStream_t stream_ = nullptr;
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() {
        if (data_) {
            (void)cudaFree(data_);
        }
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    void allocate(std::size_t bytes) {
        if (data_ || bytes == 0) {
            throw std::logic_error("DeviceBuffer requires one nonempty allocation");
        }
        check_cuda(cudaMalloc(&data_, bytes), "cudaMalloc");
    }
    void* get() const { return data_; }

private:
    void* data_ = nullptr;
};

// The ONNX output is fixed by the export contract, but TensorRT 10.3 may
// conservatively report -1 before execution. Never use -1 for allocation:
// provide a bounded buffer, then require the notified shape to match.
class FixedOutputAllocator final : public nvinfer1::IOutputAllocator {
public:
    void bind(void* memory, std::size_t capacity, int rows, int hidden) {
        memory_ = memory;
        capacity_ = capacity;
        rows_ = rows;
        hidden_ = hidden;
    }

    void reset() noexcept {
        allocation_failed_ = false;
        shape_received_ = false;
        shape_valid_ = false;
    }

    void* reallocateOutputAsync(const char* name, void* /*current_memory*/,
                               std::uint64_t size, std::uint64_t alignment,
                               cudaStream_t /*stream*/) noexcept override {
        if (!name || std::strcmp(name, kOutputName) != 0 || !memory_ || size == 0 ||
            size > capacity_ || alignment == 0 ||
            reinterpret_cast<std::uintptr_t>(memory_) % alignment != 0) {
            allocation_failed_ = true;
            return nullptr;
        }
        return memory_;
    }

    void notifyShape(const char* name, const nvinfer1::Dims& dims) noexcept override {
        shape_received_ = true;
        shape_valid_ = name && std::strcmp(name, kOutputName) == 0 &&
                       dims.nbDims == 2 && dims.d[0] == rows_ && dims.d[1] == hidden_;
    }

    void validate() const {
        if (allocation_failed_) {
            throw std::runtime_error("TensorRT output allocation exceeds the fixed export contract or has invalid alignment");
        }
        if (!shape_received_ || !shape_valid_) {
            throw std::runtime_error("TensorRT runtime output shape is missing or differs from (patches/4, hidden_size)");
        }
    }

private:
    void* memory_ = nullptr;
    std::size_t capacity_ = 0;
    int rows_ = 0;
    int hidden_ = 0;
    bool allocation_failed_ = false;
    bool shape_received_ = false;
    bool shape_valid_ = false;
};

std::vector<char> read_engine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("Cannot open TensorRT engine: " + path);
    }
    const auto length = file.tellg();
    if (length <= 0) {
        throw std::runtime_error("TensorRT engine is empty or unreadable: " + path);
    }
    const auto size = static_cast<std::uintmax_t>(static_cast<std::streamoff>(length));
    if (size > std::numeric_limits<std::size_t>::max() ||
        size > static_cast<std::uintmax_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error("TensorRT engine is too large to read: " + path);
    }
    std::vector<char> bytes(static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    if (!file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Cannot read complete TensorRT engine: " + path);
    }
    return bytes;
}

void validate_static_matrix(const nvinfer1::Dims& dims, const char* name) {
    if (dims.nbDims != 2 || dims.d[0] <= 0 || dims.d[1] <= 0 ||
        dims.d[0] > std::numeric_limits<int>::max() ||
        dims.d[1] > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(name) +
                                 " must have a positive, static 2D shape; dynamic engines are unsupported");
    }
}

bool validate_output_contract(const nvinfer1::Dims& dims, int rows, int hidden) {
    if (dims.nbDims != 2 || (dims.d[0] != -1 && dims.d[0] != rows) ||
        (dims.d[1] != -1 && dims.d[1] != hidden)) {
        throw std::runtime_error("vision_embeds shape differs from the fixed (patches/4, hidden_size) export contract");
    }
    return dims.d[0] == -1 || dims.d[1] == -1;
}

std::size_t element_size(nvinfer1::DataType dtype) {
    switch (dtype) {
    case nvinfer1::DataType::kFLOAT: return sizeof(float);
    case nvinfer1::DataType::kHALF: return sizeof(__half);
    default: throw std::runtime_error("Vision engine I/O must use FP32 or FP16");
    }
}

} // namespace

struct VitEngine::Impl {
    // Destruction order keeps the logger and runtime alive for their dependents.
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    CudaStream stream;
    DeviceBuffer device_input;
    DeviceBuffer device_output;
    FixedOutputAllocator output_allocator;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<__half> half_input;
    std::vector<__half> half_output;
    nvinfer1::DataType input_type = nvinfer1::DataType::kFLOAT;
    nvinfer1::DataType output_type = nvinfer1::DataType::kFLOAT;
    int patches = 0;
    int rows = 0;
    int hidden = 0;
    std::size_t input_elements = 0;
    std::size_t output_elements = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    bool uses_output_allocator = false;

    Impl(const std::string& path, int expected_hidden_size) {
        if (expected_hidden_size <= 0) {
            throw std::invalid_argument("Expected vision hidden size must be positive");
        }
        const auto serialized = read_engine(path);
        runtime.reset(nvinfer1::createInferRuntime(logger));
        if (!runtime) {
            throw std::runtime_error("TensorRT createInferRuntime failed");
        }
        engine.reset(runtime->deserializeCudaEngine(serialized.data(), serialized.size()));
        if (!engine) {
            throw std::runtime_error("TensorRT engine deserialization failed; check engine/device/runtime compatibility");
        }
        if (engine->getNbIOTensors() != 2) {
            throw std::runtime_error("Expected exactly pixel_values input and vision_embeds output; use the static export");
        }
        bool found_input = false;
        bool found_output = false;
        for (int i = 0; i < engine->getNbIOTensors(); ++i) {
            const char* name = engine->getIOTensorName(i);
            if (!name) {
                throw std::runtime_error("TensorRT returned a null I/O tensor name");
            }
            if (std::strcmp(name, kInputName) == 0 &&
                engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                found_input = true;
            } else if (std::strcmp(name, kOutputName) == 0 &&
                       engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
                found_output = true;
            } else {
                throw std::runtime_error(std::string("Unexpected vision engine I/O tensor: ") + name);
            }
            if (engine->getTensorLocation(name) != nvinfer1::TensorLocation::kDEVICE ||
                engine->getTensorFormat(name) != nvinfer1::TensorFormat::kLINEAR) {
                throw std::runtime_error(std::string(name) + " requires a linear, device-resident tensor");
            }
        }
        if (!found_input || !found_output) {
            throw std::runtime_error("Vision engine is missing the expected input or output");
        }
        const auto input_dims = engine->getTensorShape(kInputName);
        const auto output_dims = engine->getTensorShape(kOutputName);
        validate_static_matrix(input_dims, kInputName);
        patches = static_cast<int>(input_dims.d[0]);
        rows = patches / kMergeArea;
        hidden = expected_hidden_size;
        if (input_dims.d[1] != kPatchDim || patches % kMergeArea != 0) {
            throw std::runtime_error("Vision engine violates patch_dim=1176 and 2x2 spatial-merge shape contract");
        }
        (void)validate_output_contract(output_dims, rows, hidden);
        input_type = engine->getTensorDataType(kInputName);
        output_type = engine->getTensorDataType(kOutputName);
        input_elements = checked_product(static_cast<std::size_t>(patches), kPatchDim);
        output_elements = checked_product(static_cast<std::size_t>(rows), static_cast<std::size_t>(hidden));
        input_bytes = checked_product(input_elements, element_size(input_type));
        output_bytes = checked_product(output_elements, element_size(output_type));
        context.reset(engine->createExecutionContext());
        if (!context) {
            throw std::runtime_error("TensorRT createExecutionContext failed");
        }
        // Validate known dimensions. Unresolved output axes are checked by
        // IOutputAllocator against the same explicit contract after enqueue.
        const auto resolved_input = context->getTensorShape(kInputName);
        const auto resolved_output = context->getTensorShape(kOutputName);
        validate_static_matrix(resolved_input, kInputName);
        uses_output_allocator = validate_output_contract(resolved_output, rows, hidden);
        for (int axis = 0; axis < 2; ++axis) {
            if (resolved_input.d[axis] != input_dims.d[axis]) {
                throw std::runtime_error("Execution context input shape differs from the static engine contract");
            }
        }
        stream.create();
        device_input.allocate(input_bytes);
        device_output.allocate(output_bytes);
        output_allocator.bind(device_output.get(), output_bytes, rows, hidden);
        if (uses_output_allocator && !context->setOutputAllocator(kOutputName, &output_allocator)) {
            throw std::runtime_error("TensorRT setOutputAllocator failed");
        }
        if (!context->setTensorAddress(kInputName, device_input.get()) ||
            !context->setTensorAddress(kOutputName, device_output.get())) {
            throw std::runtime_error("TensorRT setTensorAddress failed");
        }
        if (input_type == nvinfer1::DataType::kHALF) {
            half_input.resize(input_elements);
        }
        if (output_type == nvinfer1::DataType::kHALF) {
            half_output.resize(output_elements);
        }
    }

    ~Impl() {
        // Complete outstanding work before host/device buffers are destroyed.
        if (stream.get()) {
            (void)cudaStreamSynchronize(stream.get());
        }
    }

    VisionFeatures infer(const VisionInput& input) {
        if (input.grid.t != 1 || input.grid.h <= 0 || input.grid.w <= 0 ||
            input.grid.h % 2 != 0 || input.grid.w % 2 != 0 || input.patch_dim != kPatchDim) {
            throw std::invalid_argument("VisionInput requires one image, positive even grid dimensions and patch_dim=1176");
        }
        const auto grid_patches = checked_product(static_cast<std::size_t>(input.grid.h),
                                                  static_cast<std::size_t>(input.grid.w));
        if (grid_patches != static_cast<std::size_t>(patches) || input.pixels.size() != input_elements) {
            throw std::invalid_argument("VisionInput shape does not match the static TensorRT engine");
        }
        const void* host_input = input.pixels.data();
        if (input_type == nvinfer1::DataType::kHALF) {
            for (std::size_t i = 0; i < input_elements; ++i) {
                half_input[i] = __float2half_rn(input.pixels[i]);
            }
            host_input = half_input.data();
        }
        VisionFeatures result;
        result.rows = rows;
        result.hidden_size = hidden;
        result.values.resize(output_elements);
        void* host_output = output_type == nvinfer1::DataType::kHALF
                                ? static_cast<void*>(half_output.data())
                                : static_cast<void*>(result.values.data());
        try {
            output_allocator.reset();
            check_cuda(cudaMemcpyAsync(device_input.get(), host_input, input_bytes,
                                       cudaMemcpyHostToDevice, stream.get()), "Vision input copy");
            if (!context->enqueueV3(stream.get())) {
                throw std::runtime_error("TensorRT enqueueV3 failed");
            }
            if (uses_output_allocator) {
                // TensorRT guarantees notifyShape before enqueueV3 returns.
                output_allocator.validate();
            }
            check_cuda(cudaMemcpyAsync(host_output, device_output.get(), output_bytes,
                                       cudaMemcpyDeviceToHost, stream.get()), "Vision output copy");
            check_cuda(cudaStreamSynchronize(stream.get()), "Vision stream synchronization");
        } catch (...) {
            // Keep the local output and caller's input alive until pending DMA
            // has completed, including failures after a successful enqueue.
            (void)cudaStreamSynchronize(stream.get());
            throw;
        }
        if (output_type == nvinfer1::DataType::kHALF) {
            for (std::size_t i = 0; i < output_elements; ++i) {
                result.values[i] = __half2float(half_output[i]);
            }
        }
        return result;
    }
};

VitEngine::VitEngine(const std::string& engine_path, int expected_hidden_size)
    : impl_(std::make_unique<Impl>(engine_path, expected_hidden_size)) {}
VitEngine::~VitEngine() = default;
int VitEngine::input_patches() const { return impl_->patches; }
int VitEngine::output_rows() const { return impl_->rows; }
int VitEngine::hidden_size() const { return impl_->hidden; }
VisionFeatures VitEngine::infer(const VisionInput& input) { return impl_->infer(input); }

} // namespace qwen_vl
