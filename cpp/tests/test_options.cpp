#include "qwen_vl/options.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool ok) { if (!ok) throw std::runtime_error("Option check failed"); }
template <class F> void must_throw(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw);
}
}

int main() {
    const std::filesystem::path file = "qwen_options_test.json";
    try {
        const std::vector<std::string> required = {"--vision-engine", "vit.engine", "--vision-contract", "shape.json",
            "--model", "model.gguf", "--image", "image.jpg", "--prompt", "Describe this."};
        const auto defaults = qwen_vl::parse_options(required);
        require(defaults.llama.max_new_tokens == 256 && defaults.llama.n_ctx == 4096);
        require(qwen_vl::parse_options({"--help"}).help);
        must_throw([] { qwen_vl::parse_options({}); });
        must_throw([] { qwen_vl::parse_options({"--unknown", "1"}); });
        must_throw([] { qwen_vl::parse_options({"--model"}); });
        for (const auto& number : {"-1", "1.5", "2147483648", "1x"}) {
            auto args = required;
            args.insert(args.end(), {"--max-new-tokens", number});
            must_throw([&] { qwen_vl::parse_options(args); });
        }
        {
            std::ofstream out(file);
            out << R"({"vision_engine":"vit.engine","vision_contract":"shape.json","model":"model.gguf","image":"image.jpg","prompt":"Config question","max_new_tokens":12})";
        }
        const auto overridden = qwen_vl::parse_options({"--max-new-tokens", "0", "--config", file.string(), "--prompt", "CLI question"});
        require(overridden.llama.max_new_tokens == 0 && overridden.prompt == "CLI question");
        { std::ofstream out(file); out << R"({"max_new_tokens":1.5})"; }
        must_throw([&] { qwen_vl::parse_options({"--config", file.string()}); });
        std::filesystem::remove(file);
        std::cout << "Options and config validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
