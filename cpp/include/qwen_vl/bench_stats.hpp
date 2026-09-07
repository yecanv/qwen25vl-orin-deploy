#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace qwen_vl {

struct Distribution {
    std::size_t count = 0;
    std::optional<double> minimum;
    std::optional<double> maximum;
    std::optional<double> mean;
    std::optional<double> p50;
    std::optional<double> p95;
    std::optional<double> p99;
};

struct RequestSample {
    bool success = false;
    int http_status = 0;
    std::string transport_error;
    std::string response_error;
    std::string backend;
    std::uint64_t generated_tokens = 0;
    double elapsed_ms = 0;
    std::optional<double> queue_ms;
    std::optional<double> inference_ms;
    std::optional<double> server_first_token_ms;
};

struct BenchmarkStats {
    std::size_t total = 0;
    std::size_t succeeded = 0;
    std::size_t failed = 0;
    std::uint64_t generated_tokens = 0;
    double wall_seconds = 0;
    double requests_per_second = 0;
    double successful_requests_per_second = 0;
    double generated_tokens_per_second = 0;
    std::map<int, std::size_t> http_status_counts;
    std::map<std::string, std::size_t> transport_error_counts;
    std::map<std::string, std::size_t> response_error_counts;
    std::map<std::string, std::size_t> backend_counts;
    Distribution latency_ms;
    Distribution successful_latency_ms;
    Distribution server_queue_ms;
    Distribution server_inference_ms;
    Distribution server_first_token_ms;
};

// Percentiles linearly interpolate between sorted samples at (n - 1) * p.
Distribution summarize_distribution(std::vector<double> samples);
BenchmarkStats summarize_benchmark(const std::vector<RequestSample>& samples,
                                   double wall_seconds);

}  // namespace qwen_vl
