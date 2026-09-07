#include "qwen_vl/io.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STBI_ONLY_PNM
#include "stb_image.h"
#include "json.hpp"

#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace qwen_vl {

Image read_image(const std::string& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Cannot open image: " + path);
    const auto length = stream.tellg();
    constexpr std::streamoff max_encoded_bytes = 128 * 1024 * 1024;
    if (length <= 0 || length > max_encoded_bytes)
        throw std::runtime_error("Image file must be nonempty and <= 128 MiB");
    std::vector<stbi_uc> encoded(static_cast<std::size_t>(length));
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char*>(encoded.data()), length))
        throw std::runtime_error("Cannot read complete image: " + path);
    return decode_image(encoded);
}

Image decode_image(const std::vector<std::uint8_t>& encoded) {
    if (encoded.empty() || encoded.size() > 128U * 1024U * 1024U)
        throw std::invalid_argument("Encoded image must contain 1..128 MiB");
    int width = 0, height = 0, channels = 0;
    const int size = static_cast<int>(encoded.size());
    if (!stbi_info_from_memory(encoded.data(), size, &width, &height, &channels))
        throw std::invalid_argument("Unsupported or invalid encoded image");
    constexpr std::int64_t max_pixels = 64 * 1024 * 1024;
    if (width <= 0 || height <= 0 || static_cast<std::int64_t>(width) * height > max_pixels)
        throw std::invalid_argument("Decoded image must contain 1..64M pixels");
    using PixelPtr = std::unique_ptr<stbi_uc, decltype(&stbi_image_free)>;
    PixelPtr data(stbi_load_from_memory(encoded.data(), size, &width, &height,
                                       &channels, 3), stbi_image_free);
    if (!data) throw std::invalid_argument("Image decoding failed");
    Image result;
    result.width = width;
    result.height = height;
    result.rgb.assign(data.get(), data.get() + static_cast<std::size_t>(width) * height * 3);
    return result;
}

EngineContract read_engine_contract(const std::string& path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("Cannot open engine shape contract: " + path);
    const auto root = nlohmann::json::parse(stream);
    if (!root.at("static").is_boolean() || !root.at("static").get<bool>())
        throw std::runtime_error("Only the static ViT export is supported");
    auto checked_int = [](const nlohmann::json& value) {
        if (!value.is_number_integer()) throw std::runtime_error("Shape entries must be integers");
        const auto v = value.get<std::int64_t>();
        if (v <= 0 || v > 100000000) throw std::runtime_error("Invalid positive shape dimension");
        return static_cast<int>(v);
    };
    const auto& g = root.at("grid_thw");
    const auto& in = root.at("input").at("pixel_values");
    const auto& out = root.at("output").at("vision_embeds");
    if (!g.is_array() || g.size() != 3 || !in.is_array() || in.size() != 2 ||
        !out.is_array() || out.size() != 2)
        throw std::runtime_error("Expected grid[3], pixel_values[2], vision_embeds[2]");
    EngineContract contract;
    contract.grid = {checked_int(g[0]), checked_int(g[1]), checked_int(g[2])};
    contract.patch_dim = checked_int(root.at("patch_dim"));
    contract.hidden_size = checked_int(out[1]);
    const auto patches = static_cast<std::int64_t>(contract.grid.h) * contract.grid.w;
    if (contract.grid.t != 1 || contract.grid.h % 2 || contract.grid.w % 2 ||
        contract.grid.h > 4096 || contract.grid.w > 4096 ||
        contract.patch_dim != 1176 || checked_int(in[1]) != 1176 ||
        patches != checked_int(in[0]) || patches / 4 != checked_int(out[0]))
        throw std::runtime_error("Inconsistent single-image Qwen2.5-VL static shape contract");
    contract.target_width = contract.grid.w * 14;
    contract.target_height = contract.grid.h * 14;
    return contract;
}

void write_pixels(const std::string& path, const VisionInput& input) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("Cannot create pixel output: " + path);
    stream.write(reinterpret_cast<const char*>(input.pixels.data()),
                 static_cast<std::streamsize>(input.pixels.size() * sizeof(float)));
    stream.close();
    if (!stream) throw std::runtime_error("Failed to write pixel output: " + path);
}

}  // namespace qwen_vl
