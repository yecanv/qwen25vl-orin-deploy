#include "qwen_vl/preprocess.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

qwen_vl::Image pattern(int width, int height) {
    qwen_vl::Image image;
    image.width = width;
    image.height = height;
    image.rgb.resize(static_cast<std::size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                image.rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c] =
                    static_cast<std::uint8_t>((x * 17 + y * 29 + c * 71 + (x * y % 31) * 3) % 256);
            }
        }
    }
    return image;
}

std::size_t offset(int width, int x, int y, int channel, int frame) {
    const int block = (y / 28) * (width / 28) + x / 28;
    const int patch = block * 4 + (y % 28 / 14) * 2 + x % 28 / 14;
    return static_cast<std::size_t>(patch) * 1176 + channel * 392 + frame * 196 +
           (y % 14) * 14 + x % 14;
}

constexpr std::array<float, 3> mean = {0.48145466F, 0.4578275F, 0.40821073F};
constexpr std::array<float, 3> stddev = {0.26862954F, 0.26130258F, 0.27577711F};

float normalize(int pixel, int channel) {
    return (static_cast<float>(pixel * (1.0 / 255.0)) - mean[channel]) / stddev[channel];
}

std::uint64_t rgb_hash(const qwen_vl::VisionInput& input, int width, int height) {
    require(input.grid.t == 1 && input.grid.h == height / 14 && input.grid.w == width / 14,
            "incorrect grid dimensions");
    require(input.patch_dim == 1176, "incorrect flattened patch size");
    require(input.pixels.size() == static_cast<std::size_t>(width) * height * 6,
            "incorrect tensor size");
    std::uint64_t hash = UINT64_C(14695981039346656037);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const float value = input.pixels.at(offset(width, x, y, c, 0));
                require(std::isfinite(value), "non-finite normalized pixel");
                const long pixel = std::lround((value * stddev[c] + mean[c]) * 255.0F);
                require(pixel >= 0 && pixel <= 255, "normalized pixel outside uint8 range");
                require(std::abs(value - normalize(static_cast<int>(pixel), c)) < 2e-6F,
                        "incorrect CLIP normalization");
                require(value == input.pixels.at(offset(width, x, y, c, 1)),
                        "still-image temporal duplication mismatch");
                hash = (hash ^ static_cast<std::uint8_t>(pixel)) * UINT64_C(1099511628211);
            }
        }
    }
    return hash;
}

template <class Exception, class Function>
void expect_throw(Function function) {
    bool caught = false;
    try {
        function();
    } catch (const Exception&) {
        caught = true;
    }
    require(caught, "invalid input did not throw the expected exception");
}

void test_layout() {
    const auto image = pattern(84, 56);
    const auto result = qwen_vl::preprocess(image, {84, 56});
    for (int y = 0; y < image.height; ++y) {
        for (int x = 0; x < image.width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const auto rgb_index = (static_cast<std::size_t>(y) * image.width + x) * 3 + c;
                const float expected = normalize(image.rgb[rgb_index], c);
                for (int frame = 0; frame < 2; ++frame) {
                    require(std::abs(result.pixels.at(offset(image.width, x, y, c, frame)) -
                                     expected) < 2e-6F,
                            "merge-block, channel, temporal or intra-patch order mismatch");
                }
            }
        }
    }
}

void test_pillow_resize() {
    struct Case { int width, height, out_width, out_height; std::uint64_t golden; };
    // RGB FNV-1a golden values generated independently using Pillow 11.0.0
    // Image.resize(..., Resampling.BICUBIC), including edge and mixed-axis cases.
    constexpr Case cases[] = {
        {11, 17, 28, 28, UINT64_C(0x7e2202d45fb22470)},
        {97, 65, 56, 28, UINT64_C(0x8da07ac3fdccc26b)},
        {56, 19, 56, 28, UINT64_C(0x0ffd34ee467810e1)},
        {29, 56, 56, 56, UINT64_C(0xaa41ec1d852a6e6a)},
        {300, 17, 28, 56, UINT64_C(0xd14a6fc9ba697373)},
        {1, 33, 28, 28, UINT64_C(0xa4494f13566ec795)},
        {33, 1, 28, 28, UINT64_C(0x88115212eefbf015)},
    };
    for (const auto& test : cases) {
        const auto result = qwen_vl::preprocess(pattern(test.width, test.height),
                                                {test.out_width, test.out_height});
        require(rgb_hash(result, test.out_width, test.out_height) == test.golden,
                "resized RGB differs from the Pillow golden image");
    }
}

void test_default_and_errors() {
    qwen_vl::Image pixel{1, 1, {0, 127, 255}};
    const auto result = qwen_vl::preprocess(pixel);
    require(result.grid.h == 64 && result.grid.w == 64 && result.grid.t == 1,
            "default image must be 896 x 896 with grid 1 x 64 x 64");
    (void)rgb_hash(result, 896, 896);
    for (int c = 0; c < 3; ++c) {
        require(result.pixels.at(offset(896, 895, 895, c, 1)) == normalize(pixel.rgb[c], c),
                "constant image not preserved through resize");
    }
    expect_throw<std::invalid_argument>([] { qwen_vl::preprocess({}); });
    expect_throw<std::invalid_argument>([] { qwen_vl::preprocess({-1, 1, {}}); });
    expect_throw<std::invalid_argument>([] { qwen_vl::preprocess({1, 1, {0, 0}}); });
    expect_throw<std::invalid_argument>([] { qwen_vl::preprocess({1, 1, {0, 0, 0, 0}}); });
    expect_throw<std::invalid_argument>([&] { qwen_vl::preprocess(pixel, {0, 28}); });
    expect_throw<std::invalid_argument>([&] { qwen_vl::preprocess(pixel, {28, -28}); });
    expect_throw<std::invalid_argument>([&] { qwen_vl::preprocess(pixel, {14, 28}); });
    const int enormous = std::numeric_limits<int>::max() / 28 * 28;
    expect_throw<std::length_error>([&] { qwen_vl::preprocess(pixel, {enormous, enormous}); });
}

}  // namespace

int main(int argc, char** argv) {
    try {
        // Optional integration oracle: no models or GPU required. This writes
        // native float32 data for compare_preprocess.py on the same machine.
        if (argc == 7 && std::string(argv[1]) == "--dump") {
            const int width = std::stoi(argv[3]);
            const int height = std::stoi(argv[4]);
            const int out_width = std::stoi(argv[5]);
            const int out_height = std::stoi(argv[6]);
            require(width > 0 && height > 0 && width <= 4096 && height <= 4096,
                    "test image dimensions must lie in [1, 4096]");
            require(out_width > 0 && out_height > 0 && out_width <= 4096 && out_height <= 4096,
                    "test target dimensions must lie in [1, 4096]");
            const auto result = qwen_vl::preprocess(pattern(width, height), {out_width, out_height});
            std::ofstream stream(argv[2], std::ios::binary);
            stream.write(reinterpret_cast<const char*>(result.pixels.data()),
                         static_cast<std::streamsize>(result.pixels.size() * sizeof(float)));
            require(static_cast<bool>(stream), "unable to write oracle tensor");
            return 0;
        }
        require(argc == 1, "usage: test_preprocess [--dump FILE WIDTH HEIGHT OUT_WIDTH OUT_HEIGHT]");
        test_layout();
        test_pillow_resize();
        test_default_and_errors();
        std::cout << "preprocess: layout, Pillow resize, normalization and validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "preprocess: " << error.what() << '\n';
        return 1;
    }
}
