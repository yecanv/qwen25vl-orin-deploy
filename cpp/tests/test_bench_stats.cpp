#include "qwen_vl/bench_stats.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void near(double actual, double expected) {
    require(std::abs(actual - expected) < 1e-10, "distribution or throughput mismatch");
}

template <typename Function>
void expect_invalid(Function function) {
    try { function(); } catch (const std::invalid_argument&) { return; }
    throw std::runtime_error("invalid benchmark input was accepted");
}
}  // namespace

int main() {
    try {
        const auto empty = qwen_vl::summarize_distribution({});
        require(empty.count == 0 && !empty.mean && !empty.p50 && !empty.p95 && !empty.p99,
                "empty percentiles must be absent");
        const auto singleton = qwen_vl::summarize_distribution({7});
        near(*singleton.minimum, 7);
        near(*singleton.p99, 7);
        const auto distribution = qwen_vl::summarize_distribution({30, 0, 20, 10});
        require(distribution.count == 4, "sample count is wrong");
        near(*distribution.minimum, 0);
        near(*distribution.maximum, 30);
        near(*distribution.mean, 15);
        near(*distribution.p50, 15);
        near(*distribution.p95, 28.5);
        near(*distribution.p99, 29.7);
        const double largest = std::numeric_limits<double>::max();
        require(std::isfinite(*qwen_vl::summarize_distribution({largest, largest}).mean),
                "mean must not overflow from intermediate summation");
        expect_invalid([] { qwen_vl::summarize_distribution({-1}); });
        expect_invalid([] { qwen_vl::summarize_distribution({std::numeric_limits<double>::infinity()}); });
        expect_invalid([] { qwen_vl::summarize_distribution({std::numeric_limits<double>::quiet_NaN()}); });

        qwen_vl::RequestSample success;
        success.success = true;
        success.http_status = 200;
        success.elapsed_ms = 100;
        success.generated_tokens = 8;
        success.backend = "synthetic";
        success.queue_ms = 5;
        success.inference_ms = 90;
        success.server_first_token_ms = 35;
        qwen_vl::RequestSample overloaded;
        overloaded.http_status = 429;
        overloaded.elapsed_ms = 10;
        overloaded.response_error = "http_status";
        overloaded.generated_tokens = 1000;
        overloaded.queue_ms = 900;
        qwen_vl::RequestSample connection;
        connection.elapsed_ms = 30;
        connection.transport_error = "Connection";
        const auto stats = qwen_vl::summarize_benchmark({success, overloaded, connection}, 2);
        require(stats.total == 3 && stats.succeeded == 1 && stats.failed == 2,
                "requests were double-counted or dropped");
        require(stats.http_status_counts.at(200) == 1 && stats.http_status_counts.at(429) == 1 &&
                    stats.http_status_counts.count(0) == 0 &&
                    stats.transport_error_counts.at("Connection") == 1,
                "HTTP and transport classification mismatch");
        require(stats.backend_counts.at("synthetic") == 1 && stats.generated_tokens == 8,
                "failed requests must not contribute tokens");
        near(stats.requests_per_second, 1.5);
        near(stats.successful_requests_per_second, 0.5);
        near(stats.generated_tokens_per_second, 4);
        near(*stats.latency_ms.p50, 30);
        near(*stats.successful_latency_ms.p50, 100);
        near(*stats.server_queue_ms.p50, 5);
        near(*stats.server_first_token_ms.p50, 35);
        const auto failed = qwen_vl::summarize_benchmark({overloaded, connection}, 1);
        require(failed.succeeded == 0 && !failed.successful_latency_ms.mean &&
                    !failed.server_first_token_ms.p95, "all-failed distributions must be absent");
        expect_invalid([&] { qwen_vl::summarize_benchmark({success}, 0); });
        expect_invalid([&] { qwen_vl::summarize_benchmark({success}, -1); });
        success.queue_ms = -1;
        expect_invalid([&] { qwen_vl::summarize_benchmark({success}, 1); });
        std::cout << "Benchmark statistics checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Benchmark statistics checks failed: " << error.what() << '\n';
        return 1;
    }
}
