#include "qwen_vl/bench_stats.hpp"

#include <httplib.h>
#include <json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

struct Options {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::string image;
    std::string prompt = "Describe this image.";
    std::vector<std::size_t> concurrency{1, 2, 4, 8};
    std::size_t requests = 24;
    std::size_t warmup = 2;
    int timeout_ms = 30000;
    int max_new_tokens = 128;
    std::string output;
    bool allow_errors = false;
    bool help = false;
};

std::size_t integer(const std::string& text, std::size_t minimum, std::size_t maximum,
                    const std::string& name) {
    if (text.empty()) throw std::invalid_argument(name + " requires an integer");
    std::size_t value = 0;
    for (char character : text) {
        if (character < '0' || character > '9') {
            throw std::invalid_argument(name + " requires an unsigned decimal integer");
        }
        const auto digit = static_cast<std::size_t>(character - '0');
        if (digit > maximum || value > (maximum - digit) / 10) {
            throw std::invalid_argument(name + " exceeds " + std::to_string(maximum));
        }
        value = value * 10 + digit;
    }
    if (value < minimum) throw std::invalid_argument(name + " is below " + std::to_string(minimum));
    return value;
}

std::vector<std::size_t> concurrency_list(const std::string& text) {
    std::vector<std::size_t> result;
    std::set<std::size_t> seen;
    std::size_t start = 0;
    while (true) {
        const auto end = text.find(',', start);
        const auto count = end == std::string::npos ? text.size() - start : end - start;
        const auto value = integer(text.substr(start, count), 1, 128, "--concurrency");
        if (!seen.insert(value).second) throw std::invalid_argument("duplicate concurrency value");
        if (result.size() == 16) throw std::invalid_argument("at most 16 concurrency levels are allowed");
        result.push_back(value);
        if (end == std::string::npos) break;
        start = end + 1;
    }
    return result;
}

Options parse_options(int argc, char** argv) {
    Options options;
    std::set<std::string> seen;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (!seen.insert(argument).second) throw std::invalid_argument("duplicate option: " + argument);
        if (argument == "--help" || argument == "-h") {
            options.help = true;
            continue;
        }
        if (argument == "--allow-errors") {
            options.allow_errors = true;
            continue;
        }
        if (argument != "--host" && argument != "--port" && argument != "--image" &&
            argument != "--prompt" && argument != "--concurrency" && argument != "--requests" &&
            argument != "--warmup" && argument != "--timeout-ms" &&
            argument != "--max-new-tokens" && argument != "--output") {
            throw std::invalid_argument("unknown option: " + argument);
        }
        if (index + 1 == argc) throw std::invalid_argument("missing value for " + argument);
        const std::string value = argv[++index];
        if (argument == "--host") options.host = value;
        else if (argument == "--port") options.port = static_cast<int>(integer(value, 1, 65535, argument));
        else if (argument == "--image") options.image = value;
        else if (argument == "--prompt") options.prompt = value;
        else if (argument == "--concurrency") options.concurrency = concurrency_list(value);
        else if (argument == "--requests") options.requests = integer(value, 1, 100000, argument);
        else if (argument == "--warmup") options.warmup = integer(value, 0, 10000, argument);
        else if (argument == "--timeout-ms") options.timeout_ms = static_cast<int>(integer(value, 1, 600000, argument));
        else if (argument == "--max-new-tokens") options.max_new_tokens = static_cast<int>(integer(value, 1, 32768, argument));
        else if (argument == "--output") options.output = value;
    }
    if (options.help) return options;
    if (options.host.empty() || options.host.size() > 253 || options.host.find("://") != std::string::npos ||
        options.host.find_first_of("/\\ \t\r\n@?#") != std::string::npos) {
        throw std::invalid_argument("--host must be a hostname or IP address without a URL scheme or path");
    }
    if (options.image.empty()) throw std::invalid_argument("--image is required");
    if (options.prompt.empty() || options.prompt.size() > 65536) {
        throw std::invalid_argument("--prompt must contain 1..65536 bytes");
    }
    return options;
}

std::string load_image(const std::string& path) {
    std::ifstream file(std::filesystem::u8path(path), std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open image: " + path);
    const auto length = file.tellg();
    constexpr std::streamoff max_image_bytes = 8 * 1024 * 1024;
    if (length <= 0 || length > max_image_bytes) throw std::runtime_error("image must contain 1..8388608 bytes");
    std::string bytes(static_cast<std::size_t>(length), '\0');
    file.seekg(0);
    file.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file || file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        throw std::runtime_error("failed to read complete image");
    }
    return bytes;
}

std::string base64(const std::string& bytes) {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(((bytes.size() + 2) / 3) * 4);
    for (std::size_t index = 0; index < bytes.size(); index += 3) {
        const auto a = static_cast<unsigned char>(bytes[index]);
        const auto b = index + 1 < bytes.size() ? static_cast<unsigned char>(bytes[index + 1]) : 0;
        const auto c = index + 2 < bytes.size() ? static_cast<unsigned char>(bytes[index + 2]) : 0;
        result.push_back(alphabet[a >> 2]);
        result.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
        result.push_back(index + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
        result.push_back(index + 2 < bytes.size() ? alphabet[c & 63] : '=');
    }
    return result;
}

void configure_client(httplib::Client& client, const Options& options) {
    const auto timeout = std::chrono::milliseconds(options.timeout_ms);
    client.set_connection_timeout(std::min(timeout, std::chrono::milliseconds(5000)));
    client.set_read_timeout(timeout);
    client.set_write_timeout(timeout);
    client.set_max_timeout(timeout);
    client.set_keep_alive(true);
    client.set_follow_location(false);
    client.set_decompress(false);
}

std::optional<double> timing(const Json& values, const char* name, bool nullable) {
    const auto& value = values.at(name);
    if (nullable && value.is_null()) return std::nullopt;
    if (!value.is_number()) throw std::runtime_error("timing value is not numeric");
    const double number = value.get<double>();
    if (!std::isfinite(number) || number < 0) throw std::runtime_error("timing must be finite and nonnegative");
    return number;
}

qwen_vl::RequestSample request(httplib::Client& client, const std::string& body,
                             int max_new_tokens) {
    qwen_vl::RequestSample sample;
    const auto begin = Clock::now();
    try {
        // Abort oversized responses before retaining an unbounded response body.
        std::string response_body;
        constexpr std::size_t response_limit = 8 * 1024 * 1024;
        bool response_too_large = false;
        httplib::Request outgoing;
        outgoing.method = "POST";
        outgoing.path = "/v1/generate";
        outgoing.body = body;
        // httplib 0.20.1 initializes this in Post(), but not in low-level send().
        outgoing.start_time_ = begin;
        outgoing.set_header("Content-Type", "application/json");
        outgoing.response_handler = [&](const httplib::Response& response) {
            sample.http_status = response.status;
            return true;
        };
        outgoing.content_receiver = [&](const char* data, std::size_t length,
                                        std::uint64_t, std::uint64_t) {
            if (length > response_limit - response_body.size()) {
                response_too_large = true;
                return false;
            }
            response_body.append(data, length);
            return true;
        };
        const auto response = client.send(outgoing);
        if (!response) {
            if (response_too_large) sample.response_error = "response_too_large";
            else sample.transport_error = httplib::to_string(response.error());
        } else {
            sample.http_status = response->status;
            if (response->status != 200) {
                sample.response_error = "http_status";
            } else {
                try {
                    const auto value = Json::parse(response_body);
                    const auto& id = value.at("id");
                    const auto& backend = value.at("backend");
                    const auto& tokens = value.at("generated_tokens");
                    if ((!id.is_string() && !id.is_number_unsigned()) || !value.at("text").is_string() ||
                        !backend.is_string() || backend.get_ref<const std::string&>().empty() ||
                        !tokens.is_number_unsigned() || tokens.get<std::uint64_t>() > static_cast<std::uint64_t>(max_new_tokens)) {
                        throw std::runtime_error("invalid generation response schema");
                    }
                    sample.backend = backend.get<std::string>();
                    sample.generated_tokens = tokens.get<std::uint64_t>();
                    const auto& values = value.at("timing");
                    sample.queue_ms = timing(values, "queue_ms", false);
                    sample.inference_ms = timing(values, "inference_ms", false);
                    sample.server_first_token_ms = timing(values, "first_token_ms", true);
                    sample.success = true;
                } catch (const std::exception&) {
                    sample.response_error = "invalid_response";
                }
            }
        }
    } catch (const std::exception&) {
        sample.transport_error = "client_exception";
    }
    sample.elapsed_ms = std::chrono::duration<double, std::milli>(Clock::now() - begin).count();
    return sample;
}

struct Run {
    qwen_vl::BenchmarkStats stats;
    std::size_t effective_concurrency = 0;
};

Run run(const Options& options, const std::string& body, std::size_t total,
        std::size_t concurrency) {
    if (total == 0) return {qwen_vl::summarize_benchmark({}, 0), 0};
    const auto workers = std::min(total, concurrency);
    std::vector<qwen_vl::RequestSample> samples(total);
    std::vector<std::thread> threads;
    threads.reserve(workers);
    std::atomic<std::size_t> next{0};
    std::mutex mutex;
    std::condition_variable condition;
    std::size_t ready = 0;
    bool start = false;
    bool abort = false;
    std::exception_ptr worker_error;
    auto join = [&] {
        for (auto& thread : threads) if (thread.joinable()) thread.join();
    };
    try {
        for (std::size_t worker = 0; worker < workers; ++worker) {
            threads.emplace_back([&] {
                try {
                    httplib::Client client(options.host, options.port);
                    configure_client(client, options);
                    {
                        std::unique_lock<std::mutex> lock(mutex);
                        ++ready;
                        condition.notify_all();
                        condition.wait(lock, [&] { return start || abort; });
                        if (abort) return;
                    }
                    while (true) {
                        const auto index = next.fetch_add(1, std::memory_order_relaxed);
                        if (index >= total) break;
                        samples[index] = request(client, body, options.max_new_tokens);
                    }
                } catch (...) {
                    std::lock_guard<std::mutex> lock(mutex);
                    if (!worker_error) worker_error = std::current_exception();
                    abort = true;
                    condition.notify_all();
                }
            });
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            abort = true;
        }
        condition.notify_all();
        join();
        throw;
    }
    Clock::time_point begin;
    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] { return ready == workers || abort; });
        begin = Clock::now();
        start = true;
    }
    condition.notify_all();
    join();
    const auto wall_seconds = std::chrono::duration<double>(Clock::now() - begin).count();
    if (worker_error) std::rethrow_exception(worker_error);
    return {qwen_vl::summarize_benchmark(samples, wall_seconds), workers};
}

Json nullable(const std::optional<double>& value) {
    return value ? Json(*value) : Json(nullptr);
}

Json distribution_json(const qwen_vl::Distribution& values) {
    return {{"count", values.count}, {"min", nullable(values.minimum)},
            {"max", nullable(values.maximum)}, {"mean", nullable(values.mean)},
            {"p50", nullable(values.p50)}, {"p95", nullable(values.p95)}, {"p99", nullable(values.p99)}};
}

Json run_json(const Run& run) {
    const auto& values = run.stats;
    Json status_counts = Json::object();
    for (const auto& entry : values.http_status_counts) status_counts[std::to_string(entry.first)] = entry.second;
    return {{"effective_concurrency", run.effective_concurrency},
            {"requests", values.total}, {"succeeded", values.succeeded}, {"failed", values.failed},
            {"generated_tokens", values.generated_tokens}, {"wall_seconds", values.wall_seconds},
            {"requests_per_second", values.requests_per_second},
            {"successful_requests_per_second", values.successful_requests_per_second},
            {"generated_tokens_per_second", values.generated_tokens_per_second},
            {"http_status_counts", status_counts},
            {"transport_error_counts", values.transport_error_counts},
            {"response_error_counts", values.response_error_counts},
            {"backend_counts", values.backend_counts},
            {"synthetic_backend_observed", values.backend_counts.count("synthetic") != 0},
            {"latency_ms", distribution_json(values.latency_ms)},
            {"successful_latency_ms", distribution_json(values.successful_latency_ms)},
            {"server_queue_ms", distribution_json(values.server_queue_ms)},
            {"server_inference_ms", distribution_json(values.server_inference_ms)},
            {"server_first_token_ms", distribution_json(values.server_first_token_ms)}};
}

void save(const std::string& destination, const std::string& serialized) {
    const auto path = std::filesystem::u8path(destination);
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("cannot open output file: " + destination);
    output << serialized << '\n';
    output.close();
    if (!output) throw std::runtime_error("failed to write output file: " + destination);
}

}  // namespace

int bench_main(int argc, char** argv) {
    try {
        const auto options = parse_options(argc, argv);
        if (options.help) {
            std::cout << "Usage: vlm_bench --image FILE [options]\n"
                         "  --host HOST                 Default: 127.0.0.1 (plain HTTP)\n"
                         "  --port PORT                 Default: 8080\n"
                         "  --prompt TEXT               Default: Describe this image.\n"
                         "  --concurrency 1,2,4,8       1..128 workers; at most 16 levels\n"
                         "  --requests N                Exact measured total per level; default: 24\n"
                         "  --warmup N                  Separate sequential requests per level; default: 2\n"
                         "  --timeout-ms N              Request wall timeout; default: 30000\n"
                         "  --max-new-tokens N          Default: 128\n"
                         "  --output FILE               Also save JSON output\n"
                         "  --allow-errors              Allow partial failures (all-failed still fails)\n"
                         "  --help                      Show this help\n";
            return 0;
        }
        if (!options.output.empty() && std::filesystem::exists(std::filesystem::u8path(options.output)) &&
            std::filesystem::equivalent(std::filesystem::u8path(options.image), std::filesystem::u8path(options.output))) {
            throw std::invalid_argument("--output must not overwrite the input image");
        }
        const auto bytes = load_image(options.image);
        const std::string body = Json{{"prompt", options.prompt}, {"image_base64", base64(bytes)},
            {"max_new_tokens", options.max_new_tokens}, {"timeout_ms", options.timeout_ms}}.dump();
        Json result = {{"schema", "qwen_vl_http_benchmark/v1"},
                       {"endpoint", {{"host", options.host}, {"port", options.port}, {"path", "/v1/generate"}}},
                       {"mode", "non_streaming"},
                       {"server_first_token_timing_scope", "server_submission_to_first_token"},
                       {"latency_timing_scope", "client_post_start_to_complete_response"},
                       {"percentile_method", "linear_interpolation_rank_(n-1)*p"},
                       {"requests_per_level", options.requests}, {"warmup_per_level", options.warmup},
                       {"timeout_ms", options.timeout_ms}, {"max_new_tokens", options.max_new_tokens},
                       {"image_bytes", bytes.size()}, {"allow_errors", options.allow_errors},
                       {"levels", Json::array()}};
        bool any_failure = false;
        bool all_failed_level = false;
        bool synthetic = false;
        std::set<std::string> backends;
        for (const auto concurrency : options.concurrency) {
            const auto warmup = run(options, body, options.warmup, 1);
            const auto measured = run(options, body, options.requests, concurrency);
            any_failure = any_failure || warmup.stats.failed != 0 || measured.stats.failed != 0;
            all_failed_level = all_failed_level || measured.stats.succeeded == 0;
            for (const auto* stats : {&warmup.stats, &measured.stats}) {
                for (const auto& backend : stats->backend_counts) backends.insert(backend.first);
                synthetic = synthetic || stats->backend_counts.count("synthetic") != 0;
            }
            result["levels"].push_back({{"requested_concurrency", concurrency},
                                         {"warmup", run_json(warmup)}, {"measurement", run_json(measured)}});
        }
        result["observed_backends"] = backends;
        result["synthetic_backend_observed"] = synthetic;
        result["has_failures"] = any_failure;
        const auto serialized = result.dump(2);
        if (!options.output.empty()) save(options.output, serialized);
        std::cout << serialized << '\n';
        return all_failed_level || (any_failure && !options.allow_errors) ? 2 : 0;
    } catch (const std::exception& error) {
        std::cerr << "vlm_bench: " << error.what() << '\n';
        return 1;
    }
}

#ifdef _WIN32
int wmain(int argc, wchar_t** argv) {
    try {
        std::vector<std::string> arguments;
        arguments.reserve(static_cast<std::size_t>(argc));
        for (int index = 0; index < argc; ++index) {
            const int length = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS,
                                                   argv[index], -1, nullptr, 0, nullptr, nullptr);
            if (length == 0) throw std::runtime_error("invalid Unicode command-line argument");
            std::string value(static_cast<std::size_t>(length), '\0');
            if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, argv[index], -1,
                                    value.data(), length, nullptr, nullptr) != length) {
                throw std::runtime_error("cannot convert command-line argument to UTF-8");
            }
            value.pop_back();
            arguments.push_back(std::move(value));
        }
        std::vector<char*> pointers;
        pointers.reserve(arguments.size());
        for (auto& argument : arguments) pointers.push_back(argument.data());
        return bench_main(argc, pointers.data());
    } catch (const std::exception& error) {
        std::cerr << "vlm_bench: " << error.what() << '\n';
        return 1;
    }
}
#else
int main(int argc, char** argv) { return bench_main(argc, argv); }
#endif
