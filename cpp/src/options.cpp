#include "qwen_vl/options.hpp"
#include "json.hpp"

#include <algorithm>
#include <charconv>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace qwen_vl {
namespace {
const std::map<std::string, std::string> flag_keys = {
    {"--vision-engine", "vision_engine"}, {"--vision-contract", "vision_contract"},
    {"--model", "model"}, {"--image", "image"}, {"--prompt", "prompt"},
    {"--metrics", "metrics"}, {"--ctx-size", "ctx_size"}, {"--batch-size", "batch_size"},
    {"--gpu-layers", "gpu_layers"}, {"--max-new-tokens", "max_new_tokens"}, {"--threads", "threads"}
};
const std::set<std::string> integer_keys = {"ctx_size", "batch_size", "gpu_layers", "max_new_tokens", "threads"};

int integer(const std::string& text) {
    int number = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), number);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
        throw std::runtime_error("Expected an integer, got: " + text);
    return number;
}

void assign(RuntimeOptions& result, const std::string& key, const std::string& value) {
    if (key == "vision_engine") result.vision_engine = value;
    else if (key == "vision_contract") result.vision_contract = value;
    else if (key == "model") result.llama.model_path = value;
    else if (key == "image") result.image = value;
    else if (key == "prompt") result.prompt = value;
    else if (key == "metrics") result.metrics = value;
    else if (key == "ctx_size") result.llama.n_ctx = integer(value);
    else if (key == "batch_size") result.llama.n_batch = integer(value);
    else if (key == "gpu_layers") result.llama.n_gpu_layers = integer(value);
    else if (key == "max_new_tokens") result.llama.max_new_tokens = integer(value);
    else if (key == "threads") result.llama.n_threads = integer(value);
    else throw std::runtime_error("Unknown config key: " + key);
}
}

RuntimeOptions parse_options(const std::vector<std::string>& args) {
    RuntimeOptions result;
    if (std::find(args.begin(), args.end(), "--help") != args.end()) {
        result.help = true;
        return result;
    }
    std::map<std::string, std::string> flags;
    for (std::size_t i = 0; i < args.size(); i += 2) {
        const auto& name = args[i];
        if (name != "--config" && !flag_keys.count(name)) throw std::runtime_error("Unknown argument: " + name);
        if (i + 1 >= args.size() || flags.count(name)) throw std::runtime_error("Missing or repeated argument: " + name);
        flags[name] = args[i + 1];
    }
    if (flags.count("--config")) {
        std::ifstream file(flags.at("--config"));
        if (!file) throw std::runtime_error("Cannot open config: " + flags.at("--config"));
        const auto config = nlohmann::json::parse(file);
        if (!config.is_object()) throw std::runtime_error("Config must be a JSON object");
        for (auto it = config.begin(); it != config.end(); ++it) {
            if (integer_keys.count(it.key())) {
                if (!it.value().is_number_integer()) throw std::runtime_error("Config requires integer: " + it.key());
                // Parse textual representation to reject values outside signed int.
                assign(result, it.key(), it.value().dump());
            } else {
                if (!it.value().is_string()) throw std::runtime_error("Config requires string: " + it.key());
                assign(result, it.key(), it.value().get<std::string>());
            }
        }
    }
    for (const auto& entry : flags) {
        if (entry.first != "--config") assign(result, flag_keys.at(entry.first), entry.second);
    }
    if (result.vision_engine.empty() || result.vision_contract.empty() || result.llama.model_path.empty() ||
        result.image.empty() || result.prompt.empty())
        throw std::runtime_error("Required: --vision-engine, --vision-contract, --model, --image, --prompt (or --config)");
    if (result.llama.n_ctx <= 0 || result.llama.n_batch <= 0 || result.llama.n_gpu_layers < 0 ||
        result.llama.max_new_tokens < 0 || result.llama.n_threads < 0)
        throw std::runtime_error("Context/batch must be positive; GPU layers/new tokens/threads must be nonnegative");
    return result;
}

std::string usage() {
    return "vlm_cli --config cpp/configs/single_image.json [overrides]\n"
           "Required: --vision-engine FILE --vision-contract SHAPES.json --model GGUF --image FILE --prompt TEXT\n"
           "Optional: --ctx-size 4096 --batch-size 2048 --gpu-layers 99 --max-new-tokens 256 --threads 0 --metrics FILE\n"
           "Single image, fixed engine resolution, greedy decoding. Paths are relative to the working directory.\n";
}
}  // namespace qwen_vl
