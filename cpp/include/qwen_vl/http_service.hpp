#pragma once

#include "qwen_vl/service.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace qwen_vl::service {

std::vector<std::uint8_t> decode_base64(const std::string& text, std::size_t max_bytes);
std::string encode_base64(const std::vector<std::uint8_t>& bytes);

struct HttpOptions {
    std::string host = "127.0.0.1";
    int port = 8080;
    std::size_t http_threads = 16;
    std::size_t connection_queue = 32;
    std::size_t queue_capacity = 8;
    std::size_t workers = 1;
    int max_tokens = 512;
    int default_timeout_ms = 30000;
    int max_timeout_ms = 120000;
    int socket_timeout_ms = 5000;
    std::size_t max_image_bytes = 8 * 1024 * 1024;
};

class HttpService {
public:
    HttpService(Scheduler& scheduler, HttpOptions options, std::string backend_name);
    ~HttpService();
    HttpService(const HttpService&) = delete;
    HttpService& operator=(const HttpService&) = delete;

    bool listen();
    int bind_to_any_port();
    bool listen_after_bind();
    bool is_running() const;
    void stop();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace qwen_vl::service
