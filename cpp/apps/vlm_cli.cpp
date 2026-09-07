#include "qwen_vl/io.hpp"
#include "qwen_vl/llama_decoder.hpp"
#include "qwen_vl/options.hpp"
#include "qwen_vl/preprocess.hpp"
#include "qwen_vl/vit_engine.hpp"
#include "json.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}

int main(int argc, char** argv) {
    try {
        const auto options = qwen_vl::parse_options(std::vector<std::string>(argv + 1, argv + argc));
        if (options.help) {
            std::cout << qwen_vl::usage();
            return 0;
        }
        const auto contract = qwen_vl::read_engine_contract(options.vision_contract);
        const auto load_start = Clock::now();
        qwen_vl::VitEngine vision(options.vision_engine, contract.hidden_size);
        if (static_cast<std::int64_t>(contract.grid.h) * contract.grid.w != vision.input_patches())
            throw std::runtime_error("Engine input does not match the selected shape contract");
        qwen_vl::LlamaDecoder decoder(options.llama);
        const double load_ms = elapsed(load_start);

        // Request timing starts before file decode/preprocessing; model loading is separate.
        const auto request_start = Clock::now();
        const auto image = qwen_vl::read_image(options.image);
        const auto input = qwen_vl::preprocess(image, {contract.target_width, contract.target_height});
        const double preprocess_ms = elapsed(request_start);
        const auto vision_start = Clock::now();
        const auto features = vision.infer(input);
        const double vision_ms = elapsed(vision_start);
        double first_output_ms = -1.0;
        const auto result = decoder.generate(features, input.grid, options.prompt, [&](const std::string& piece) {
            std::cout << piece << std::flush;
            if (first_output_ms < 0.0 && !piece.empty()) first_output_ms = elapsed(request_start);
        });
        const double request_ms = elapsed(request_start);
        std::cout << '\n';
        nlohmann::json metrics = {
            {"schema_version", 1}, {"backend", "tensorrt_vit_llamacpp"},
            {"load_ms", load_ms}, {"preprocess_ms", preprocess_ms}, {"vision_ms", vision_ms},
            {"context_init_ms", result.timing.context_init_ms}, {"prefill_ms", result.timing.prefill_ms},
            {"llm_first_token_ms", result.timing.first_token_ms >= 0.0 ? nlohmann::json(result.timing.first_token_ms) : nlohmann::json(nullptr)},
            {"request_first_output_ms", first_output_ms >= 0.0 ? nlohmann::json(first_output_ms) : nlohmann::json(nullptr)},
            {"request_ms", request_ms}, {"generation_ms", result.timing.generation_ms},
            {"generated_tokens", result.generated_tokens}, {"decoded_tokens", result.decoded_tokens},
            {"stopped_on_eog", result.stopped_on_eog},
            {"grid_thw", {input.grid.t, input.grid.h, input.grid.w}},
            {"vision_shape", {features.rows, features.hidden_size}},
            {"timing_note", "request starts before image file decoding; model loading excluded; first_output includes stdout flush"}
        };
        std::cerr << "TIMING_JSON " << metrics.dump() << '\n';
        if (!options.metrics.empty()) {
            std::ofstream file(options.metrics);
            if (!file) throw std::runtime_error("Cannot create metrics: " + options.metrics);
            file << metrics.dump(2) << '\n';
            file.close();
            if (!file) throw std::runtime_error("Failed to write metrics: " + options.metrics);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
