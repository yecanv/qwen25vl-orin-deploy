#include "qwen_vl/llama_decoder.hpp"
#include "qwen_vl/mrope.hpp"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qwen_vl {
namespace {

using Clock = std::chrono::steady_clock;

double elapsed_ms(Clock::time_point start, Clock::time_point end = Clock::now()) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::mutex backend_mutex;
std::size_t backend_users = 0;

// Backend lifetime must outlive every model and context, including on exceptions.
class BackendGuard {
public:
    BackendGuard() {
        const std::lock_guard<std::mutex> lock(backend_mutex);
        if (backend_users == 0) {
            ggml_backend_load_all();
            llama_backend_init();
        }
        ++backend_users;
    }
    ~BackendGuard() {
        const std::lock_guard<std::mutex> lock(backend_mutex);
        if (--backend_users == 0) llama_backend_free();
    }
    BackendGuard(const BackendGuard&) = delete;
    BackendGuard& operator=(const BackendGuard&) = delete;
};

using ModelPtr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using ContextPtr = std::unique_ptr<llama_context, decltype(&llama_free)>;
using SamplerPtr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

LlamaOptions validate_options(LlamaOptions options) {
    if (options.model_path.empty()) throw std::invalid_argument("GGUF model_path is required");
    if (options.n_ctx <= 0 || options.n_batch <= 0 || options.max_new_tokens < 0 ||
        options.n_threads < 0 || options.n_gpu_layers < -1) {
        throw std::invalid_argument("invalid llama context, batch, token, thread, or GPU-layer limit");
    }
    if (options.max_new_tokens >= options.n_ctx) {
        throw std::invalid_argument("max_new_tokens must leave context space for the image and prompt");
    }
    return options;
}

std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text,
                                  bool parse_special) {
    if (text.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
        throw std::length_error("tokenizer input exceeds int32 length");
    }
    if (text.empty()) return {};
    const auto length = static_cast<std::int32_t>(text.size());
    const int needed = llama_tokenize(vocab, text.data(), length, nullptr, 0, false, parse_special);
    if (needed == std::numeric_limits<int>::min()) {
        throw std::length_error("token count exceeds tokenizer capacity");
    }
    if (needed == 0) return {};
    if (needed > 0) throw std::runtime_error("unexpected tokenizer sizing result");
    std::vector<llama_token> tokens(static_cast<std::size_t>(-needed));
    const int count = llama_tokenize(vocab, text.data(), length, tokens.data(), -needed,
                                     false, parse_special);
    if (count < 0 || count > -needed) throw std::runtime_error("tokenizer buffer size changed");
    tokens.resize(static_cast<std::size_t>(count));
    return tokens;
}

std::string token_piece(const llama_vocab* vocab, llama_token token) {
    std::vector<char> buffer(128);
    int count = llama_token_to_piece(vocab, token, buffer.data(),
                                     static_cast<int>(buffer.size()), 0, false);
    if (count == std::numeric_limits<int>::min()) {
        throw std::length_error("token piece exceeds int32 capacity");
    }
    if (count < 0) {
        buffer.resize(static_cast<std::size_t>(-count));
        count = llama_token_to_piece(vocab, token, buffer.data(),
                                     static_cast<int>(buffer.size()), 0, false);
    }
    if (count < 0 || static_cast<std::size_t>(count) > buffer.size()) {
        throw std::runtime_error("token-to-piece conversion failed");
    }
    return {buffer.data(), static_cast<std::size_t>(count)};
}

// Own the arrays with vectors rather than relying on unchecked C allocations.
// Text positions are scalar: llama.cpp broadcasts them across its M-RoPE sections.
class TextBatch {
public:
    explicit TextBatch(std::size_t capacity)
        : tokens_(capacity), positions_(capacity), sequence_counts_(capacity, 1),
          sequences_(capacity + 1U, nullptr), logits_(capacity, 0) {
        for (std::size_t i = 0; i < capacity; ++i) sequences_[i] = &sequence_zero_;
    }

    void decode(llama_context* context, const llama_token* tokens, std::size_t count,
                llama_pos start, bool logits_last) {
        if (count == 0 || count > tokens_.size()) {
            throw std::logic_error("invalid internal text batch size");
        }
        for (std::size_t i = 0; i < count; ++i) {
            tokens_[i] = tokens[i];
            positions_[i] = start + static_cast<llama_pos>(i);
            logits_[i] = static_cast<std::int8_t>(logits_last && i + 1U == count);
        }
        llama_batch batch{};
        batch.n_tokens = static_cast<std::int32_t>(count);
        batch.token = tokens_.data();
        batch.pos = positions_.data();
        batch.n_seq_id = sequence_counts_.data();
        batch.seq_id = sequences_.data();
        batch.logits = logits_.data();
        const int status = llama_decode(context, batch);
        if (status != 0) {
            throw std::runtime_error("llama text decode failed with status " + std::to_string(status));
        }
    }

    void decode_all(llama_context* context, const std::vector<llama_token>& tokens,
                    llama_pos start, bool logits_last, const std::function<bool()>& stop_requested) {
        for (std::size_t offset = 0; offset < tokens.size();) {
            if (stop_requested && stop_requested()) throw std::runtime_error("Generation cancelled");
            const auto count = std::min(tokens_.size(), tokens.size() - offset);
            decode(context, tokens.data() + offset, count,
                   start + static_cast<llama_pos>(offset),
                   logits_last && offset + count == tokens.size());
            offset += count;
        }
    }

private:
    llama_seq_id sequence_zero_ = 0;
    std::vector<llama_token> tokens_;
    std::vector<llama_pos> positions_;
    std::vector<std::int32_t> sequence_counts_;
    std::vector<llama_seq_id*> sequences_;
    std::vector<std::int8_t> logits_;
};

void decode_image(llama_context* context, const VisionFeatures& features, ImageMrope& rope) {
    const auto n = static_cast<std::size_t>(features.rows);
    std::vector<std::int32_t> sequence_counts(n, 1);
    llama_seq_id sequence_zero = 0;
    std::vector<llama_seq_id*> sequences(n + 1U, &sequence_zero);
    sequences.back() = nullptr;
    std::vector<std::int8_t> logits(n, 0);
    llama_batch batch{};
    batch.n_tokens = features.rows;
    // llama_decode reads this input; its public C batch struct has no const pointer.
    batch.embd = const_cast<float*>(features.values.data());
    batch.pos = rope.positions.data();
    batch.n_seq_id = sequence_counts.data();
    batch.seq_id = sequences.data();
    batch.logits = logits.data();
    const int status = llama_decode(context, batch);
    if (status != 0) {
        throw std::runtime_error("llama image decode failed with status " + std::to_string(status));
    }
}

}  // namespace

struct LlamaDecoder::Impl {
    LlamaOptions options;
    BackendGuard backend;
    ModelPtr model{nullptr, llama_model_free};
    const llama_vocab* vocab = nullptr;
    int hidden_size = 0;
    double model_load_ms = 0.0;

    explicit Impl(LlamaOptions supplied) : options(validate_options(std::move(supplied))) {
        const auto start = Clock::now();
        auto params = llama_model_default_params();
        params.n_gpu_layers = options.n_gpu_layers;
        model.reset(llama_model_load_from_file(options.model_path.c_str(), params));
        if (!model) throw std::runtime_error("cannot load GGUF model: " + options.model_path);
        if (llama_model_rope_type(model.get()) != LLAMA_ROPE_TYPE_MROPE) {
            throw std::invalid_argument("expected a Qwen2.5-VL GGUF with M-RoPE");
        }
        vocab = llama_model_get_vocab(model.get());
        hidden_size = llama_model_n_embd_inp(model.get());
        if (!vocab || hidden_size <= 0) throw std::runtime_error("GGUF has no usable vocabulary/embedding");
        for (const std::string marker : {"<|im_start|>", "<|im_end|>",
                                         "<|vision_start|>", "<|vision_end|>"}) {
            const auto token = tokenize(vocab, marker, true);
            if (token.size() != 1U) throw std::invalid_argument("GGUF lacks Qwen marker " + marker);
        }
        model_load_ms = elapsed_ms(start);
    }
};

LlamaDecoder::LlamaDecoder(LlamaOptions options) : impl_(std::make_unique<Impl>(std::move(options))) {}
LlamaDecoder::~LlamaDecoder() = default;
LlamaDecoder::LlamaDecoder(LlamaDecoder&&) noexcept = default;
LlamaDecoder& LlamaDecoder::operator=(LlamaDecoder&&) noexcept = default;

GenerationResult LlamaDecoder::generate(
    const VisionFeatures& features, const Grid& grid, const std::string& question,
    const std::function<void(const std::string&)>& on_piece, const GenerationControl& control) {
    if (!impl_) throw std::logic_error("cannot use a moved-from LlamaDecoder");
    const auto start = Clock::now();
    const auto& state = *impl_;
    auto options = state.options;
    if (control.max_new_tokens >= 0) options.max_new_tokens = control.max_new_tokens;
    if (control.max_new_tokens < -1 || options.max_new_tokens >= options.n_ctx)
        throw std::invalid_argument("Invalid per-request max_new_tokens");
    const auto check_stop = [&] {
        if (control.stop_requested && control.stop_requested()) throw std::runtime_error("Generation cancelled");
    };
    check_stop();
    if (features.rows <= 0 || features.hidden_size != state.hidden_size) {
        throw std::invalid_argument("vision feature rows/hidden_size do not match the GGUF input");
    }
    // Validate the grid before allocating 4*rows positions. A mismatched caller-supplied
    // grid must not cause a large allocation merely to discover a feature mismatch.
    if (grid.t != 1 || grid.h <= 0 || grid.w <= 0 || grid.h % 2 != 0 || grid.w % 2 != 0 ||
        static_cast<std::int64_t>(grid.h / 2) * (grid.w / 2) != features.rows) {
        throw std::invalid_argument("vision rows do not match a single-image 2x2-merged patch grid");
    }
    const auto expected = static_cast<std::uint64_t>(features.rows) * features.hidden_size;
    if (expected != features.values.size()) throw std::invalid_argument("vision feature buffer size mismatch");
    if (!std::all_of(features.values.begin(), features.values.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("vision features contain NaN or infinity");
    }
    if (question.empty()) throw std::invalid_argument("question must not be empty");

    GenerationResult result;
    result.timing.model_load_ms = state.model_load_ms;
    const auto tokenize_start = Clock::now();
    const auto prefix = tokenize(state.vocab,
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<|vision_start|>", true);
    auto suffix = tokenize(state.vocab, "<|vision_end|>", true);
    const auto question_tokens = tokenize(state.vocab, question, false);
    suffix.insert(suffix.end(), question_tokens.begin(), question_tokens.end());
    const auto answer_prefix = tokenize(state.vocab, "<|im_end|>\n<|im_start|>assistant\n", true);
    suffix.insert(suffix.end(), answer_prefix.begin(), answer_prefix.end());
    result.timing.tokenize_ms = elapsed_ms(tokenize_start);
    if (prefix.empty() || suffix.empty()) throw std::runtime_error("Qwen template tokenized to an empty segment");

    const auto prompt_count = static_cast<std::uint64_t>(prefix.size()) + features.rows + suffix.size();
    const auto required = prompt_count + static_cast<std::uint64_t>(options.max_new_tokens);
    if (required > static_cast<std::uint64_t>(options.n_ctx)) {
        throw std::invalid_argument("prompt + image + output requires " + std::to_string(required) +
                                    " context tokens; increase n_ctx or reduce image/output size");
    }
    auto rope = make_image_mrope(grid, static_cast<llama_pos>(prefix.size()));
    if (rope.rows != features.rows) throw std::invalid_argument("vision rows do not match the merged patch grid");

    const auto context_start = Clock::now();
    check_stop();
    auto params = llama_context_default_params();
    params.n_ctx = static_cast<std::uint32_t>(options.n_ctx);
    params.n_batch = static_cast<std::uint32_t>(std::min(options.n_batch, options.n_ctx));
    params.n_ubatch = params.n_batch;
    params.n_seq_max = 1;
    if (options.n_threads > 0) {
        params.n_threads = options.n_threads;
        params.n_threads_batch = options.n_threads;
    }
    ContextPtr context(llama_init_from_model(state.model.get(), params), llama_free);
    if (!context) throw std::runtime_error("cannot create llama context");
    result.timing.context_init_ms = elapsed_ms(context_start);
    if (required > llama_n_ctx(context.get())) throw std::invalid_argument("actual llama context is too small");
    const auto actual_batch = llama_n_batch(context.get());
    if (actual_batch == 0 || actual_batch > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("llama returned an unsupported batch capacity");
    }
    if (static_cast<std::uint32_t>(features.rows) > actual_batch ||
        static_cast<std::uint32_t>(features.rows) > llama_n_ubatch(context.get())) {
        throw std::invalid_argument("image embeddings must fit one batch/microbatch; increase n_batch");
    }
    TextBatch batch(actual_batch);
    const auto prefill_start = Clock::now();
    batch.decode_all(context.get(), prefix, 0, false, control.stop_requested);
    check_stop();
    decode_image(context.get(), features, rope);
    check_stop();
    batch.decode_all(context.get(), suffix, rope.next_position, true, control.stop_requested);
    llama_synchronize(context.get());
    check_stop();
    result.timing.prefill_ms = elapsed_ms(prefill_start);
    if (options.max_new_tokens == 0) return result;

    SamplerPtr sampler(llama_sampler_init_greedy(), llama_sampler_free);
    if (!sampler) throw std::runtime_error("cannot create greedy sampler");
    llama_pos next_position = rope.next_position + static_cast<llama_pos>(suffix.size());
    const auto generation_start = Clock::now();
    Clock::time_point first_token_time{};
    for (int step = 0; step < options.max_new_tokens; ++step) {
        check_stop();
        const llama_token token = llama_sampler_sample(sampler.get(), context.get(), -1);
        if (token == LLAMA_TOKEN_NULL) throw std::runtime_error("sampler returned no token");
        if (llama_vocab_is_eog(state.vocab, token)) {
            result.stopped_on_eog = true;
            break;
        }
        if (result.generated_tokens == 0) {
            first_token_time = Clock::now();
            result.timing.first_token_ms = elapsed_ms(start, first_token_time);
        }
        ++result.generated_tokens;
        const auto piece = token_piece(state.vocab, token);
        result.text += piece;
        if (on_piece && !piece.empty()) on_piece(piece);
        // The final requested token need not be decoded: there is no next sample.
        if (step + 1 == options.max_new_tokens) break;
        check_stop();
        batch.decode(context.get(), &token, 1U, next_position, true);
        ++result.decoded_tokens;
        ++next_position;
    }
    llama_synchronize(context.get());
    const auto finish = Clock::now();
    result.timing.generation_ms = elapsed_ms(generation_start, finish);
    if (result.generated_tokens > 0) result.timing.decode_ms = elapsed_ms(first_token_time, finish);
    return result;
}

}  // namespace qwen_vl
