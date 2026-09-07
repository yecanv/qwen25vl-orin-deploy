#include "qwen_vl/http_service.hpp"

#include <httplib.h>
#include <json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <limits>
#include <stdexcept>
#include <utility>

namespace qwen_vl::service {
namespace {
using Json = nlohmann::json;
constexpr std::size_t max_body_bytes = 12 * 1024 * 1024;
constexpr std::size_t max_prompt_bytes = 64 * 1024;
constexpr std::size_t max_image_limit = 8 * 1024 * 1024;
std::atomic<std::uint64_t> next_request_id{1};

class ServiceServer final : public httplib::Server {
public:
    void close_listener() noexcept {
        // httplib 0.20.1 stop() only closes a listener after listen() starts.
        // Also cover bind-only destruction and repeated concurrent stop calls.
        const auto socket = svr_sock_.exchange(INVALID_SOCKET);
        if (socket != INVALID_SOCKET) {
            httplib::detail::shutdown_socket(socket);
            httplib::detail::close_socket(socket);
        }
    }
};

int base64_digit(unsigned char ch) {
    if (ch >= 'A' && ch <= 'Z') return ch - 'A';
    if (ch >= 'a' && ch <= 'z') return ch - 'a' + 26;
    if (ch >= '0' && ch <= '9') return ch - '0' + 52;
    if (ch == '+') return 62;
    if (ch == '/') return 63;
    return -1;
}

std::string request_id(httplib::Response& response) {
    if (response.has_header("X-Request-Id")) return response.get_header_value("X-Request-Id");
    auto id = "req-" + std::to_string(next_request_id.fetch_add(1, std::memory_order_relaxed));
    response.set_header("X-Request-Id", id);
    response.set_header("Cache-Control", "no-store");
    return id;
}

void json_response(httplib::Response& response, int status, const Json& body) {
    response.status = status;
    response.set_content(body.dump(-1, ' ', false, Json::error_handler_t::replace),
                         "application/json; charset=utf-8");
}

void error_response(httplib::Response& response, int status, const std::string& code,
                    const std::string& message, const std::string& backend) {
    const auto id = request_id(response);
    json_response(response, status, {{"id", id}, {"backend", backend},
        {"error", {{"code", code}, {"message", message}}}});
    if (status == 429) response.set_header("Retry-After", "1");
}

int integer_option(const Json& body, const char* key, int default_value, int maximum) {
    if (!body.contains(key)) return std::min(default_value, maximum);
    const auto& value = body.at(key);
    if (!value.is_number_integer() || value < 1 || value > maximum) {
        throw std::invalid_argument(std::string(key) + " must be an integer between 1 and " +
                                    std::to_string(maximum));
    }
    return value.get<int>();
}

void check_options(const HttpOptions& options) {
    if (options.host.empty() || options.port < 0 || options.port > 65535 ||
        options.workers == 0 || options.workers > 64 || options.queue_capacity == 0 ||
        options.queue_capacity > 1024 || options.http_threads > 2048 ||
        options.http_threads < options.workers + options.queue_capacity + 2 ||
        options.connection_queue == 0 || options.connection_queue > 4096 ||
        options.max_tokens < 1 || options.max_tokens > 1048576 ||
        options.default_timeout_ms < 1 || options.max_timeout_ms < options.default_timeout_ms ||
        options.max_timeout_ms > 3600000 || options.socket_timeout_ms < 1 ||
        options.socket_timeout_ms > 3600000 || options.max_image_bytes == 0 ||
        options.max_image_bytes > max_image_limit) {
        throw std::invalid_argument("Invalid HTTP limits; reserve at least workers + queue_capacity + 2 HTTP threads");
    }
}
}  // namespace

std::vector<std::uint8_t> decode_base64(const std::string& text, std::size_t max_bytes) {
    if (text.empty()) return {};
    if (text.size() % 4 != 0) throw std::invalid_argument("image_base64 must use padded base64");
    const std::size_t padding = text.back() == '=' ? (text[text.size() - 2] == '=' ? 2 : 1) : 0;
    const std::size_t decoded_size = text.size() / 4 * 3 - padding;
    if (decoded_size > max_bytes) throw std::invalid_argument("Decoded image exceeds the byte limit");
    std::vector<std::uint8_t> bytes;
    bytes.reserve(decoded_size);
    for (std::size_t i = 0; i < text.size(); i += 4) {
        const bool last = i + 4 == text.size();
        const int a = base64_digit(static_cast<unsigned char>(text[i]));
        const int b = base64_digit(static_cast<unsigned char>(text[i + 1]));
        const int c = base64_digit(static_cast<unsigned char>(text[i + 2]));
        const int d = base64_digit(static_cast<unsigned char>(text[i + 3]));
        const bool two_padding = last && text[i + 2] == '=' && text[i + 3] == '=';
        const bool one_padding = last && c >= 0 && text[i + 3] == '=';
        if (a < 0 || b < 0 || (c < 0 && !two_padding) ||
            (d < 0 && !two_padding && !one_padding) ||
            (two_padding && (b & 15) != 0) || (one_padding && (c & 3) != 0)) {
            throw std::invalid_argument("image_base64 contains invalid base64");
        }
        bytes.push_back(static_cast<std::uint8_t>((a << 2) | (b >> 4)));
        if (!two_padding) bytes.push_back(static_cast<std::uint8_t>((b << 4) | (c >> 2)));
        if (!two_padding && !one_padding) bytes.push_back(static_cast<std::uint8_t>((c << 6) | d));
    }
    return bytes;
}

std::string encode_base64(const std::vector<std::uint8_t>& bytes) {
    constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    if (bytes.size() > (std::numeric_limits<std::size_t>::max() / 4) * 3 - 2) {
        throw std::length_error("Input is too large to base64-encode");
    }
    std::string result;
    result.reserve((bytes.size() + 2) / 3 * 4);
    for (std::size_t i = 0; i < bytes.size(); i += 3) {
        const std::uint32_t a = bytes[i];
        const std::uint32_t b = i + 1 < bytes.size() ? bytes[i + 1] : 0;
        const std::uint32_t c = i + 2 < bytes.size() ? bytes[i + 2] : 0;
        result.push_back(alphabet[a >> 2]);
        result.push_back(alphabet[((a & 3) << 4) | (b >> 4)]);
        result.push_back(i + 1 < bytes.size() ? alphabet[((b & 15) << 2) | (c >> 6)] : '=');
        result.push_back(i + 2 < bytes.size() ? alphabet[c & 63] : '=');
    }
    return result;
}

struct HttpService::Impl {
    Scheduler& scheduler;
    HttpOptions options;
    std::string backend;
    ServiceServer server;
    std::atomic<bool> stopped{false};

    Impl(Scheduler& scheduler_arg, HttpOptions options_arg, std::string backend_arg)
        : scheduler(scheduler_arg), options(std::move(options_arg)), backend(std::move(backend_arg)) {
        check_options(options);
        if (backend.empty()) throw std::invalid_argument("Backend name must not be empty");
        server.new_task_queue = [this] {
            return new httplib::ThreadPool(options.http_threads, options.connection_queue);
        };
        server.set_payload_max_length(max_body_bytes);
        server.set_read_timeout(std::chrono::milliseconds(options.socket_timeout_ms));
        server.set_write_timeout(std::chrono::milliseconds(options.socket_timeout_ms));
        server.set_keep_alive_max_count(1);
        server.set_keep_alive_timeout(1);
        server.set_pre_routing_handler([](const httplib::Request&, httplib::Response& response) {
            request_id(response);
            return httplib::Server::HandlerResponse::Unhandled;
        });
        server.set_error_handler([this](const httplib::Request&, httplib::Response& response) {
            if (!response.body.empty()) return;
            error_response(response, response.status, "http_error", "HTTP request rejected", backend);
        });
        server.set_exception_handler([this](const httplib::Request&, httplib::Response& response,
                                           std::exception_ptr) {
            error_response(response, 500, "internal_error", "Request handling failed", backend);
        });
        server.Get("/health", [this](const httplib::Request&, httplib::Response& response) {
            const auto snapshot = scheduler.snapshot();
            if (!snapshot.accepting) {
                error_response(response, 503, "stopping", "Server is stopping", backend);
                return;
            }
            json_response(response, 200,
                {{"id", request_id(response)}, {"status", "ok"},
                 {"backend", backend}});
        });
        server.Get("/metrics", [this](const httplib::Request&, httplib::Response& response) {
            const auto s = scheduler.snapshot();
            json_response(response, 200, {{"id", request_id(response)}, {"backend", backend},
                {"accepted", s.accepted}, {"finished", s.finished}, {"rejected", s.rejected},
                {"timed_out", s.timed_out}, {"cancelled", s.cancelled}, {"failed", s.failed},
                {"queued", s.queued}, {"active", s.active}, {"peak_active", s.peak_active},
                {"accepting", s.accepting}});
        });
        server.Post("/v1/generate", [this](const httplib::Request& http_request, httplib::Response& response) {
            generate(http_request, response);
        });
    }

    void result_response(httplib::Response& response, const Response& result) const {
        switch (result.status) {
            case Status::ok:
                json_response(response, 200, {{"id", request_id(response)}, {"text", result.text},
                    {"backend", backend}, {"generated_tokens", result.generated_tokens},
                    {"timing", {{"queue_ms", result.queue_ms}, {"inference_ms", result.inference_ms},
                                {"first_token_ms", result.first_token_ms < 0 ? Json(nullptr) : Json(result.first_token_ms)}}}});
                return;
            case Status::queue_full:
                error_response(response, 429, "queue_full", "Inference queue is full", backend);
                return;
            case Status::timeout:
                error_response(response, 504, "timeout", "Request deadline exceeded", backend);
                return;
            case Status::cancelled:
                error_response(response, 503, "cancelled", "Request was cancelled", backend);
                return;
            case Status::stopping:
                error_response(response, 503, "stopping", "Server is stopping", backend);
                return;
            case Status::backend_error:
                error_response(response, 500, "backend_error", "Inference backend failed", backend);
                return;
            case Status::invalid_request:
                error_response(response, 400, "invalid_request", "Invalid inference request", backend);
                return;
        }
        error_response(response, 500, "internal_error", "Invalid backend response status", backend);
    }

    void generate(const httplib::Request& http_request, httplib::Response& response) {
        const auto received = Clock::now();
        Request request;
        request.id = request_id(response);
        try {
            const auto content_type = http_request.get_header_value("Content-Type");
            if (content_type != "application/json" && content_type.rfind("application/json;", 0) != 0) {
                throw std::invalid_argument("Content-Type must be application/json");
            }
            const auto body = Json::parse(http_request.body);
            if (!body.is_object()) throw std::invalid_argument("Request body must be a JSON object");
            for (auto item = body.begin(); item != body.end(); ++item) {
                if (item.key() != "prompt" && item.key() != "image_base64" &&
                    item.key() != "max_new_tokens" && item.key() != "timeout_ms") {
                    throw std::invalid_argument("Request contains an unsupported field");
                }
            }
            if (!body.contains("prompt") || !body.at("prompt").is_string()) {
                throw std::invalid_argument("prompt must be a string");
            }
            request.prompt = body.at("prompt").get<std::string>();
            if (request.prompt.empty() || request.prompt.size() > max_prompt_bytes ||
                request.prompt.find_first_not_of(" \t\r\n") == std::string::npos) {
                throw std::invalid_argument("prompt must be nonempty and at most 65536 bytes");
            }
            if (!body.contains("image_base64") || !body.at("image_base64").is_string()) {
                throw std::invalid_argument("image_base64 must be a string");
            }
            request.image = decode_base64(body.at("image_base64").get_ref<const std::string&>(),
                                          options.max_image_bytes);
            if (request.image.empty()) throw std::invalid_argument("image_base64 must not be empty");
            request.max_new_tokens = integer_option(body, "max_new_tokens", 128, options.max_tokens);
            const int timeout = integer_option(body, "timeout_ms", options.default_timeout_ms,
                                               options.max_timeout_ms);
            request.deadline = received + std::chrono::milliseconds(timeout);
        } catch (const Json::exception&) {
            error_response(response, 400, "invalid_request", "Invalid JSON request", backend);
            return;
        } catch (const std::invalid_argument& error) {
            error_response(response, 400, "invalid_request", error.what(), backend);
            return;
        }
        const auto deadline = request.deadline;
        auto submission = scheduler.submit(std::move(request));
        if (submission.status != Status::ok) {
            Response result;
            result.status = submission.status;
            result_response(response, result);
            return;
        }
        if (submission.future.wait_until(deadline) != std::future_status::ready) {
            submission.cancel();
            error_response(response, 504, "timeout", "Request deadline exceeded", backend);
            return;
        }
        result_response(response, submission.future.get());
    }
};

HttpService::HttpService(Scheduler& scheduler, HttpOptions options, std::string backend_name)
    : impl_(std::make_unique<Impl>(scheduler, std::move(options), std::move(backend_name))) {}
HttpService::~HttpService() { stop(); }
bool HttpService::listen() {
    if (impl_->stopped.load()) return false;
    if (!impl_->server.bind_to_port(impl_->options.host, impl_->options.port)) return false;
    return listen_after_bind();
}
int HttpService::bind_to_any_port() {
    if (impl_->stopped.load()) return -1;
    const auto port = impl_->server.bind_to_any_port(impl_->options.host);
    if (impl_->stopped.load()) {
        impl_->server.close_listener();
        return -1;
    }
    return port;
}
bool HttpService::listen_after_bind() {
    if (impl_->stopped.load()) {
        impl_->server.close_listener();
        return false;
    }
    return impl_->server.listen_after_bind();
}
bool HttpService::is_running() const { return impl_->server.is_running(); }
void HttpService::stop() {
    impl_->stopped.store(true);
    impl_->server.close_listener();
}

}  // namespace qwen_vl::service
