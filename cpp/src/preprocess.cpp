#include "qwen_vl/preprocess.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen_vl {
namespace {

constexpr int kPatch = 14;
constexpr int kMerge = 2;
constexpr int kTemporal = 2;
constexpr int kChannels = 3;
constexpr int kPatchDim = kChannels * kTemporal * kPatch * kPatch;
constexpr std::int64_t kCoefficientScale = std::int64_t{1} << 22;
constexpr std::array<float, 3> kMean = {0.48145466F, 0.4578275F, 0.40821073F};
constexpr std::array<float, 3> kStd = {0.26862954F, 0.26130258F, 0.27577711F};

std::size_t multiply(std::size_t a, std::size_t b) {
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b) {
        throw std::length_error("image or tensor dimensions overflow size_t");
    }
    return a * b;
}

std::size_t rgb_size(int width, int height) {
    return multiply(multiply(static_cast<std::size_t>(width),
                             static_cast<std::size_t>(height)), kChannels);
}

// Keys cubic convolution with a = -0.5. Together with antialias widening,
// renormalized edge taps, 22-bit coefficients and uint8 intermediate rounding,
// this matches Pillow's RGB BICUBIC resize without reducing_gap optimization.
// Reference: Pillow 11.0.0 src/libImaging/Resample.c (see tests for the oracle).
double cubic(double coordinate) {
    const double x = std::abs(coordinate);
    if (x < 1.0) {
        return ((1.5 * x - 2.5) * x) * x + 1.0;
    }
    if (x < 2.0) {
        return (((x - 5.0) * x + 8.0) * x - 4.0) * -0.5;
    }
    return 0.0;
}

struct Taps {
    int first = 0;
    std::vector<std::int32_t> weights;
};

std::vector<Taps> coefficients(int source, int destination) {
    // Pillow takes its source box as float, then calculates coefficients as
    // doubles. Normal camera sizes are represented exactly by this cast.
    const double scale = static_cast<double>(static_cast<float>(source)) / destination;
    const double filter_scale = std::max(1.0, scale);
    const double radius = 2.0 * filter_scale;
    std::vector<Taps> result(static_cast<std::size_t>(destination));
    for (int out = 0; out < destination; ++out) {
        const double center = (out + 0.5) * scale;
        // Clamp before casting: the unclamped support can exceed int range.
        const int begin = static_cast<int>(std::clamp(center - radius + 0.5, 0.0,
                                                      static_cast<double>(source)));
        const int end = static_cast<int>(std::clamp(center + radius + 0.5, 0.0,
                                                    static_cast<double>(source)));
        auto& taps = result[static_cast<std::size_t>(out)];
        taps.first = begin;
        taps.weights.resize(static_cast<std::size_t>(end - begin));
        std::vector<double> raw(taps.weights.size());
        double sum = 0.0;
        const double inverse = 1.0 / filter_scale;
        for (int in = begin; in < end; ++in) {
            const double weight = cubic((in - center + 0.5) * inverse);
            raw[static_cast<std::size_t>(in - begin)] = weight;
            sum += weight;
        }
        if (sum == 0.0) {
            throw std::runtime_error("bicubic filter has no contributing pixels");
        }
        for (std::size_t i = 0; i < raw.size(); ++i) {
            const double scaled = raw[i] / sum * static_cast<double>(kCoefficientScale);
            taps.weights[i] = static_cast<std::int32_t>(scaled + (scaled < 0 ? -0.5 : 0.5));
        }
    }
    return result;
}

std::uint8_t to_byte(std::int64_t accumulated) {
    // Negative values are clipped before division, so no implementation-defined
    // right shift of a negative signed integer is needed in C++17.
    return static_cast<std::uint8_t>(std::clamp<std::int64_t>(
        accumulated / kCoefficientScale, 0, 255));
}

std::vector<std::uint8_t> resize_rgb(const Image& image, int width, int height) {
    std::vector<std::uint8_t> horizontal;
    const std::vector<std::uint8_t>* intermediate = &image.rgb;
    if (width != image.width) {
        const auto taps = coefficients(image.width, width);
        horizontal.resize(rgb_size(width, image.height));
        for (int y = 0; y < image.height; ++y) {
            for (int x = 0; x < width; ++x) {
                const auto& sample = taps[static_cast<std::size_t>(x)];
                for (int channel = 0; channel < kChannels; ++channel) {
                    std::int64_t sum = kCoefficientScale / 2;
                    for (std::size_t i = 0; i < sample.weights.size(); ++i) {
                        const auto in = (static_cast<std::size_t>(y) * image.width +
                                         static_cast<std::size_t>(sample.first) + i) * kChannels + channel;
                        sum += static_cast<std::int64_t>(image.rgb[in]) * sample.weights[i];
                    }
                    const auto out = (static_cast<std::size_t>(y) * width + x) * kChannels + channel;
                    horizontal[out] = to_byte(sum);
                }
            }
        }
        intermediate = &horizontal;
    }
    if (height == image.height) {
        return *intermediate;
    }
    const auto taps = coefficients(image.height, height);
    std::vector<std::uint8_t> result(rgb_size(width, height));
    for (int y = 0; y < height; ++y) {
        const auto& sample = taps[static_cast<std::size_t>(y)];
        for (int x = 0; x < width; ++x) {
            for (int channel = 0; channel < kChannels; ++channel) {
                std::int64_t sum = kCoefficientScale / 2;
                for (std::size_t i = 0; i < sample.weights.size(); ++i) {
                    const auto in = ((static_cast<std::size_t>(sample.first) + i) * width + x) *
                                        kChannels + channel;
                    sum += static_cast<std::int64_t>((*intermediate)[in]) * sample.weights[i];
                }
                const auto out = (static_cast<std::size_t>(y) * width + x) * kChannels + channel;
                result[out] = to_byte(sum);
            }
        }
    }
    return result;
}

}  // namespace

VisionInput preprocess(const Image& image, const PreprocessOptions& options) {
    if (image.width <= 0 || image.height <= 0) {
        throw std::invalid_argument("RGB image dimensions must be positive");
    }
    if (image.rgb.size() != rgb_size(image.width, image.height)) {
        throw std::invalid_argument("RGB buffer size must equal width * height * 3");
    }
    const int width = options.target_width;
    const int height = options.target_height;
    if (width <= 0 || height <= 0 || width % (kPatch * kMerge) != 0 ||
        height % (kPatch * kMerge) != 0) {
        throw std::invalid_argument("target dimensions must be positive multiples of 28");
    }

    const auto elements = multiply(rgb_size(width, height), kTemporal);
    if (elements > std::vector<float>().max_size()) {
        throw std::length_error("preprocessed tensor exceeds vector capacity");
    }
    // Validate intermediate dimensions before allocating either image.
    (void)rgb_size(width, image.height);
    const auto resized = resize_rgb(image, width, height);

    VisionInput result;
    result.grid = {1, height / kPatch, width / kPatch};
    result.patch_dim = kPatchDim;
    result.pixels.resize(elements);
    std::array<std::array<float, 256>, kChannels> normalized{};
    for (int c = 0; c < kChannels; ++c) {
        for (int value = 0; value < 256; ++value) {
            // HF rescale uses float64 multiplication and then casts to float32.
            const float scaled = static_cast<float>(value * (1.0 / 255.0));
            normalized[c][value] = (scaled - kMean[c]) / kStd[c];
        }
    }

    // Qwen flatten order: block_y, block_x, inner_patch_y, inner_patch_x,
    // channel, temporal frame, patch_y, patch_x. A still image is duplicated
    // across the two temporal frames; grid.t remains 1.
    std::size_t out = 0;
    for (int block_y = 0; block_y < result.grid.h; block_y += kMerge) {
        for (int block_x = 0; block_x < result.grid.w; block_x += kMerge) {
            for (int inner_y = 0; inner_y < kMerge; ++inner_y) {
                for (int inner_x = 0; inner_x < kMerge; ++inner_x) {
                    for (int c = 0; c < kChannels; ++c) {
                        for (int temporal = 0; temporal < kTemporal; ++temporal) {
                            for (int py = 0; py < kPatch; ++py) {
                                const int y = (block_y + inner_y) * kPatch + py;
                                for (int px = 0; px < kPatch; ++px) {
                                    const int x = (block_x + inner_x) * kPatch + px;
                                    const auto in = (static_cast<std::size_t>(y) * width + x) *
                                                        kChannels + c;
                                    result.pixels[out++] = normalized[c][resized[in]];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return result;
}

}  // namespace qwen_vl
