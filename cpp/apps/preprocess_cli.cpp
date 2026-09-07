#include "qwen_vl/io.hpp"
#include "qwen_vl/preprocess.hpp"

#include <iostream>
#include <map>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    try {
        std::map<std::string, std::string> options;
        for (int i = 1; i < argc; ++i) {
            const std::string name = argv[i];
            if (name == "--help") {
                std::cout << "qwen_preprocess --image INPUT --vision-contract SHAPES.json --output PIXELS.f32\n";
                return 0;
            }
            if (name != "--image" && name != "--vision-contract" && name != "--output")
                throw std::runtime_error("Unknown argument: " + name);
            if (++i == argc || options.count(name)) throw std::runtime_error("Missing or repeated argument: " + name);
            options[name] = argv[i];
        }
        if (options.size() != 3) throw std::runtime_error("Use --help for required arguments");
        const auto contract = qwen_vl::read_engine_contract(options.at("--vision-contract"));
        const auto image = qwen_vl::read_image(options.at("--image"));
        const auto input = qwen_vl::preprocess(image, {contract.target_width, contract.target_height});
        qwen_vl::write_pixels(options.at("--output"), input);
        std::cout << "grid_thw=" << input.grid.t << ',' << input.grid.h << ',' << input.grid.w
                  << " pixel_values=" << input.pixels.size() / input.patch_dim << 'x' << input.patch_dim << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
