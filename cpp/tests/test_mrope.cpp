#include "qwen_vl/mrope.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template <typename Exception, typename Function>
void expect_error(Function&& function, const std::string& message) {
    try {
        function();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(message);
}
}  // namespace

int main() {
    try {
        // Hand-calculated non-square image: patch grid 4x6 -> merged grid 2x3.
        // The next text position is 10, not the physical token position 13.
        const auto rectangular = qwen_vl::make_image_mrope({1, 4, 6}, 7);
        const std::vector<std::int32_t> expected = {
            7, 7, 7, 7, 7, 7,
            7, 7, 7, 8, 8, 8,
            7, 8, 9, 7, 8, 9,
            0, 0, 0, 0, 0, 0,
        };
        require(rectangular.positions == expected, "rectangular M-RoPE coordinates differ");
        require(rectangular.rows == 6 && rectangular.next_position == 10,
                "M-RoPE must advance by max merged dimension, not token count");

        // Actual 896x896 / 14-pixel patches: 64x64 -> 1024 embeddings.
        const auto deployed = qwen_vl::make_image_mrope({1, 64, 64}, 18);
        require(deployed.rows == 1024 && deployed.next_position == 50,
                "deployed image grid must merge to 32x32");
        require(deployed.positions.at(1024 + 31) == 18 &&
                    deployed.positions.at(2048 + 31) == 49 &&
                    deployed.positions.at(1024 + 32) == 19 &&
                    deployed.positions.at(2048 + 32) == 18,
                "merged-grid row boundary is incorrect");
        const auto minimal = qwen_vl::make_image_mrope({1, 2, 2}, 0);
        require(minimal.positions == std::vector<std::int32_t>({0, 0, 0, 0}) &&
                    minimal.next_position == 1, "single merged patch is incorrect");

        expect_error<std::invalid_argument>([] { qwen_vl::make_image_mrope({2, 4, 4}, 0); },
                                            "video grid must be rejected");
        expect_error<std::invalid_argument>([] { qwen_vl::make_image_mrope({1, 3, 4}, 0); },
                                            "non-divisible grid must be rejected");
        expect_error<std::invalid_argument>([] { qwen_vl::make_image_mrope({1, 0, 4}, 0); },
                                            "empty grid must be rejected");
        expect_error<std::invalid_argument>([] { qwen_vl::make_image_mrope({1, 4, 4}, -1); },
                                            "negative start must be rejected");
        expect_error<std::invalid_argument>([] { qwen_vl::make_image_mrope({1, 4, 4}, 0, 0); },
                                            "zero merge must be rejected");
        expect_error<std::overflow_error>([] {
            qwen_vl::make_image_mrope({1, 4, 4}, std::numeric_limits<std::int32_t>::max());
        }, "position overflow must be rejected");
        expect_error<std::overflow_error>([] {
            qwen_vl::make_image_mrope({1, 100000, 100000}, 0);
        }, "token-count overflow must be rejected before allocation");
        std::cout << "M-RoPE CPU checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "M-RoPE CPU checks failed: " << error.what() << '\n';
        return 1;
    }
}
