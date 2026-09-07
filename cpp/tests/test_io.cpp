#include "qwen_vl/io.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
void require(bool value, const char* description) {
    if (!value) throw std::runtime_error(description);
}
template <class F> void must_throw(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw, "Invalid input was accepted");
}
}

int main() {
    const std::filesystem::path dir = "qwen_io_test_files";
    try {
        std::filesystem::create_directories(dir);
        const auto image_path = dir / "rgb.ppm";
        {
            std::ofstream file(image_path, std::ios::binary);
            file << "P6\n2 1\n255\n";
            const unsigned char pixels[] = {255, 0, 0, 0, 127, 255};
            file.write(reinterpret_cast<const char*>(pixels), sizeof(pixels));
        }
        const auto image = qwen_vl::read_image(image_path.string());
        require(image.width == 2 && image.height == 1 && image.rgb.size() == 6, "PPM shape");
        require(image.rgb[0] == 255 && image.rgb[4] == 127 && image.rgb[5] == 255, "RGB channel order");
        const auto contract_path = dir / "shape.json";
        const std::string good = R"({"static":true,"patch_dim":1176,"grid_thw":[1,64,64],"input":{"pixel_values":[4096,1176]},"output":{"vision_embeds":[1024,2048]}})";
        auto write_contract = [&](const std::string& content) { std::ofstream file(contract_path); file << content; };
        write_contract(good);
        const auto contract = qwen_vl::read_engine_contract(contract_path.string());
        require(contract.target_width == 896 && contract.target_height == 896 && contract.hidden_size == 2048, "Static shape contract");
        auto bad = good;
        bad.replace(bad.find("4096"), 4, "4095");
        write_contract(bad);
        must_throw([&] { qwen_vl::read_engine_contract(contract_path.string()); });
        bad = good;
        bad.replace(bad.find("true"), 4, "false");
        write_contract(bad);
        must_throw([&] { qwen_vl::read_engine_contract(contract_path.string()); });
        write_contract("{}");
        must_throw([&] { qwen_vl::read_engine_contract(contract_path.string()); });
        must_throw([&] { qwen_vl::read_image(contract_path.string()); });
        must_throw([&] { qwen_vl::read_image((dir / "does_not_exist.png").string()); });
        qwen_vl::VisionInput values;
        values.pixels = {1.0f, -2.0f, 3.5f};
        const auto output = dir / "pixels.f32";
        qwen_vl::write_pixels(output.string(), values);
        require(std::filesystem::file_size(output) == 3 * sizeof(float), "Binary output size");
        // Only delete files created by this test; never recursively delete a directory.
        for (const auto& file : {image_path, contract_path, output}) std::filesystem::remove(file);
        std::filesystem::remove(dir);
        std::cout << "Image I/O and shape validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
