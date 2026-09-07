#include "qwen_vl/service_backends.hpp"
#include "qwen_vl/io.hpp"
#include "qwen_vl/preprocess.hpp"
#include "qwen_vl/vit_engine.hpp"

#include <memory>
#include <mutex>
#include <stdexcept>

namespace qwen_vl::service {
namespace {
struct ModelState {
    EngineContract contract;
    VitEngine vision;
    LlamaDecoder decoder;
    const int max_new_tokens;
    std::mutex mutex;

    explicit ModelState(const ModelBackendOptions& options)
        : contract(read_engine_contract(options.vision_contract)),
          vision(options.vision_engine, contract.hidden_size), decoder(options.llama),
          max_new_tokens(options.llama.max_new_tokens) {
        if (static_cast<std::int64_t>(contract.grid.h) * contract.grid.w != vision.input_patches())
            throw std::invalid_argument("Vision engine and shape contract do not match");
    }
};
}

Backend make_model_backend(const ModelBackendOptions& options) {
    auto state = std::make_shared<ModelState>(options);
    return [state](const Request& request, const StopRequested& stop) {
        std::unique_lock<std::mutex> lock(state->mutex, std::try_to_lock);
        if (!lock.owns_lock()) throw std::logic_error("Model backend requires one scheduler worker");
        Response response;
        response.id = request.id;
        response.backend = "tensorrt_vit_llamacpp";
        const auto start = Clock::now();
        const auto cancelled = [&] {
            if (!stop()) return false;
            response.status = Status::cancelled;
            response.error = "Request cancelled";
            return true;
        };
        if (cancelled()) return response;
        try {
            if (request.max_new_tokens < 0 || request.max_new_tokens > state->max_new_tokens)
                throw std::invalid_argument("Requested output exceeds the model token limit");
            const auto image = decode_image(request.image);
            const auto input = preprocess(image, {state->contract.target_width, state->contract.target_height});
            if (cancelled()) return response;
            const auto features = state->vision.infer(input);
            if (cancelled()) return response;
            const auto before_llm_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
            GenerationControl control;
            control.max_new_tokens = request.max_new_tokens;
            control.stop_requested = stop;
            const auto generation = state->decoder.generate(features, input.grid, request.prompt, {}, control);
            response.text = generation.text;
            response.generated_tokens = generation.generated_tokens;
            if (generation.timing.first_token_ms >= 0.0)
                response.first_token_ms = before_llm_ms + generation.timing.first_token_ms;
        } catch (const std::invalid_argument& error) {
            if (cancelled()) return response;
            response.status = Status::invalid_request;
            response.error = error.what();
        } catch (const std::exception&) {
            if (cancelled()) return response;
            throw;
        }
        return response;
    };
}
}  // namespace qwen_vl::service
