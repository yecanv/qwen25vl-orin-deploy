#include "qwen_vl/bench_stats.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qwen_vl {
namespace {

void require_nonnegative_finite(double value) {
    if (!std::isfinite(value) || value < 0) {
        throw std::invalid_argument("benchmark values must be finite and nonnegative");
    }
}

double percentile(const std::vector<double>& sorted, double probability) {
    const double rank = static_cast<double>(sorted.size() - 1) * probability;
    const auto low = static_cast<std::size_t>(rank);
    const auto high = std::min(low + 1, sorted.size() - 1);
    return sorted[low] + (sorted[high] - sorted[low]) * (rank - static_cast<double>(low));
}

void append(const std::optional<double>& value, std::vector<double>& destination) {
    if (value) {
        require_nonnegative_finite(*value);
        destination.push_back(*value);
    }
}

double rate(double count, double seconds) {
    const double result = seconds == 0 ? 0 : count / seconds;
    require_nonnegative_finite(result);
    return result;
}

}  // namespace

Distribution summarize_distribution(std::vector<double> samples) {
    Distribution result;
    result.count = samples.size();
    if (samples.empty()) return result;
    double mean = 0;
    std::size_t count = 0;
    for (double sample : samples) {
        require_nonnegative_finite(sample);
        mean += (sample - mean) / static_cast<double>(++count);
    }
    std::sort(samples.begin(), samples.end());
    result.minimum = samples.front();
    result.maximum = samples.back();
    result.mean = mean;
    result.p50 = percentile(samples, 0.50);
    result.p95 = percentile(samples, 0.95);
    result.p99 = percentile(samples, 0.99);
    return result;
}

BenchmarkStats summarize_benchmark(const std::vector<RequestSample>& samples,
                                   double wall_seconds) {
    require_nonnegative_finite(wall_seconds);
    if (!samples.empty() && wall_seconds == 0) {
        throw std::invalid_argument("nonempty benchmark requires positive wall time");
    }
    BenchmarkStats result;
    result.total = samples.size();
    result.wall_seconds = wall_seconds;
    std::vector<double> elapsed, successful_elapsed, queued, inference, first_token;
    elapsed.reserve(samples.size());
    for (const auto& sample : samples) {
        require_nonnegative_finite(sample.elapsed_ms);
        elapsed.push_back(sample.elapsed_ms);
        if (sample.http_status != 0) ++result.http_status_counts[sample.http_status];
        if (!sample.transport_error.empty()) ++result.transport_error_counts[sample.transport_error];
        if (!sample.response_error.empty()) ++result.response_error_counts[sample.response_error];
        if (!sample.backend.empty()) ++result.backend_counts[sample.backend];
        if (sample.success) {
            ++result.succeeded;
            if (sample.generated_tokens > std::numeric_limits<std::uint64_t>::max() - result.generated_tokens) {
                throw std::overflow_error("generated token counter overflow");
            }
            result.generated_tokens += sample.generated_tokens;
            successful_elapsed.push_back(sample.elapsed_ms);
            append(sample.queue_ms, queued);
            append(sample.inference_ms, inference);
            append(sample.server_first_token_ms, first_token);
        } else {
            ++result.failed;
        }
    }
    result.requests_per_second = rate(static_cast<double>(result.total), wall_seconds);
    result.successful_requests_per_second = rate(static_cast<double>(result.succeeded), wall_seconds);
    result.generated_tokens_per_second = rate(static_cast<double>(result.generated_tokens), wall_seconds);
    result.latency_ms = summarize_distribution(std::move(elapsed));
    result.successful_latency_ms = summarize_distribution(std::move(successful_elapsed));
    result.server_queue_ms = summarize_distribution(std::move(queued));
    result.server_inference_ms = summarize_distribution(std::move(inference));
    result.server_first_token_ms = summarize_distribution(std::move(first_token));
    return result;
}

}  // namespace qwen_vl
