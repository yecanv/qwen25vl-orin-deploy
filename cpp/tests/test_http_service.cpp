#include "qwen_vl/http_service.hpp"

#include <httplib.h>
#include <json.hpp>

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace std::chrono_literals;
using namespace qwen_vl::service;
using Json = nlohmann::json;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

template<class F> void must_throw(F action) {
    bool threw = false;
    try { action(); } catch (const std::exception&) { threw = true; }
    require(threw, "Expected validation to reject the input");
}

template<class F> void wait_for(F condition) {
    const auto deadline = Clock::now() + 3s;
    while (!condition()) {
        if (Clock::now() >= deadline) throw std::runtime_error("Timed out waiting for test state");
        std::this_thread::sleep_for(1ms);
    }
}

HttpOptions options() {
    HttpOptions result;
    result.port = 0;
    result.http_threads = 4;
    result.workers = 1;
    result.queue_capacity = 1;
    result.connection_queue = 4;
    result.default_timeout_ms = 4000;
    result.max_timeout_ms = 5000;
    result.max_image_bytes = 16;
    result.max_tokens = 16;
    return result;
}

struct RunningServer {
    std::atomic<bool> release{false};
    Scheduler scheduler;
    HttpService service;
    int port;
    std::thread listener;

    RunningServer()
        : scheduler(1, 1, [this](const Request& request, const StopRequested& stop) {
              if (request.prompt == "fail") throw std::runtime_error("Synthetic failure");
              if (request.prompt == "hold") {
                  while (!release.load() && !stop()) std::this_thread::sleep_for(1ms);
              }
              Response response;
              if (request.prompt == "empty") {
                  response.backend = "synthetic";
                  return response;
              }
              response.text = "synthetic:" + request.prompt;
              response.backend = "synthetic";
              response.generated_tokens = request.max_new_tokens;
              response.first_token_ms = 1.0;
              return response;
          }), service(scheduler, options(), "synthetic"), port(service.bind_to_any_port()) {
        require(port > 0, "Could not bind test server to a loopback port");
        listener = std::thread([this] { service.listen_after_bind(); });
        try { wait_for([this] { return service.is_running(); }); }
        catch (...) {
            service.stop();
            scheduler.shutdown(false);
            listener.join();
            throw;
        }
    }

    ~RunningServer() {
        release.store(true);
        service.stop();
        scheduler.shutdown(false);
        if (listener.joinable()) listener.join();
    }

    httplib::Result post(const std::string& body, const char* content_type = "application/json") const {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(2s);
        client.set_read_timeout(6s);
        client.set_write_timeout(2s);
        return client.Post("/v1/generate", body, content_type);
    }

    httplib::Result get(const char* path) const {
        httplib::Client client("127.0.0.1", port);
        client.set_connection_timeout(2s);
        client.set_read_timeout(2s);
        return client.Get(path);
    }
};

Json request(const std::string& prompt = "hello") {
    return {{"prompt", prompt}, {"image_base64", "AAEC"}, {"max_new_tokens", 4}};
}

Json check_response(const httplib::Result& response, int status) {
    require(static_cast<bool>(response), "HTTP client did not receive a response");
    require(response->status == status, "Unexpected HTTP response status");
    const auto body = Json::parse(response->body);
    require(body.at("id") == response->get_header_value("X-Request-Id"), "Request id header mismatch");
    require(body.at("backend") == "synthetic", "Synthetic backend label must be explicit");
    if (status >= 400) {
        require(body.contains("error") && body.at("error").contains("code") && body.at("error").contains("message"),
                "Structured error must contain code and message");
    }
    return body;
}

void test_base64() {
    require(encode_base64({}) == "", "Empty base64 encoding");
    require(decode_base64("", 0).empty(), "Empty base64 decoding");
    require(encode_base64({'f'}) == "Zg==", "One-byte base64 encoding");
    require(encode_base64({'f', 'o'}) == "Zm8=", "Two-byte base64 encoding");
    require(encode_base64({'f', 'o', 'o'}) == "Zm9v", "Three-byte base64 encoding");
    std::vector<std::uint8_t> bytes;
    for (int i = 0; i < 256; ++i) bytes.push_back(static_cast<std::uint8_t>(i));
    for (int i = 0; i < 3; ++i) {
        require(decode_base64(encode_base64(bytes), bytes.size()) == bytes, "Binary base64 round-trip");
        bytes.pop_back();
    }
    for (const auto* invalid : {"Zg", "====", "Zg=Z", "Zg==AAAA", "Zh==", "Zm9=", "AA A", "AA-A", "AA_A"}) {
        must_throw([&] { decode_base64(invalid, 128); });
    }
    must_throw([] { decode_base64("AAAA", 2); });
}

void test_validation(RunningServer& server) {
    check_response(server.get("/health"), 200);
    check_response(server.get("/missing"), 404);
    check_response(server.post("{"), 400);
    check_response(server.post("[]"), 400);
    check_response(server.post(request().dump(), "text/plain"), 400);
    check_response(server.post(std::string(12 * 1024 * 1024 + 1, 'x')), 413);
    for (const auto* field : {"stream", "image_url", "image_path", "model", "temperature"}) {
        auto body = request();
        body[field] = false;
        check_response(server.post(body.dump()), 400);
    }
    for (const auto& prompt : {std::string(), std::string(" \t\n"), std::string(65537, 'x')}) {
        check_response(server.post(request(prompt).dump()), 400);
    }
    for (const auto* image : {"", "https://example.com/image.png", "AAAA=", "AB==", "data:image/png;base64,AAAA"}) {
        auto body = request();
        body["image_base64"] = image;
        check_response(server.post(body.dump()), 400);
    }
    auto too_large = request();
    too_large["image_base64"] = encode_base64(std::vector<std::uint8_t>(17, 0));
    check_response(server.post(too_large.dump()), 400);
    for (const auto* field : {"prompt", "image_base64"}) {
        auto body = request();
        body.erase(field);
        check_response(server.post(body.dump()), 400);
        body[field] = 1;
        check_response(server.post(body.dump()), 400);
    }
    for (const auto* field : {"max_new_tokens", "timeout_ms"}) {
        for (const Json& value : {Json(0), Json(-1), Json(1.5), Json(true), Json("1"), Json(2147483648ULL)}) {
            auto body = request();
            body[field] = value;
            check_response(server.post(body.dump()), 400);
        }
    }
    auto too_many_tokens = request();
    too_many_tokens["max_new_tokens"] = 17;
    check_response(server.post(too_many_tokens.dump()), 400);
    auto too_long = request();
    too_long["timeout_ms"] = 5001;
    check_response(server.post(too_long.dump()), 400);
    auto default_tokens = request();
    default_tokens.erase("max_new_tokens");
    require(check_response(server.post(default_tokens.dump()), 200).at("generated_tokens") == 16,
            "Default token limit must respect the server maximum");
    auto boundary_image = request();
    boundary_image["image_base64"] = encode_base64(std::vector<std::uint8_t>(16, 0));
    const auto response = check_response(server.post(boundary_image.dump()), 200);
    require(response.at("text") == "synthetic:hello", "Unexpected generated text");
    require(response.at("generated_tokens") == 4, "Generation limit not forwarded");
    require(response.at("timing").contains("queue_ms") && response.at("timing").contains("inference_ms") &&
            response.at("timing").contains("first_token_ms"), "Missing timing fields");
    const auto empty = check_response(server.post(request("empty").dump()), 200);
    require(empty.at("generated_tokens") == 0 && empty.at("text") == "",
            "Immediate end-of-generation should return an empty successful response");
    require(empty.at("timing").at("first_token_ms").is_null(),
            "A response without generated tokens must not report a first-token timestamp");
    const auto failed = check_response(server.post(request("fail").dump()), 500);
    require(failed.at("error").at("code") == "backend_error", "Incorrect backend error mapping");
}

void test_queue_and_health(RunningServer& server) {
    server.release.store(false);
    auto active = std::async(std::launch::async, [&] { return server.post(request("hold").dump()); });
    wait_for([&] { return server.scheduler.snapshot().active == 1; });
    auto queued = std::async(std::launch::async, [&] { return server.post(request("hold").dump()); });
    wait_for([&] { return server.scheduler.snapshot().queued == 1; });
    const auto full = check_response(server.post(request().dump()), 429);
    require(full.at("error").at("code") == "queue_full", "Queue overload code mismatch");
    check_response(server.get("/health"), 200);
    const auto metrics = check_response(server.get("/metrics"), 200);
    require(metrics.at("active") == 1 && metrics.at("queued") == 1, "Metrics must expose saturation");
    require(metrics.at("peak_active") == 1, "Worker concurrency limit violated");
    server.release.store(true);
    const auto first = check_response(active.get(), 200);
    const auto second = check_response(queued.get(), 200);
    require(first.at("id") != second.at("id"), "Concurrent requests need distinct ids");
    wait_for([&] { return server.scheduler.snapshot().active == 0; });
}

void test_deadline_and_stop(RunningServer& server) {
    server.release.store(false);
    auto body = request("hold");
    body["timeout_ms"] = 40;
    const auto timed_out = check_response(server.post(body.dump()), 504);
    require(timed_out.at("error").at("code") == "timeout", "Deadline HTTP status mismatch");
    wait_for([&] { return server.scheduler.snapshot().active == 0; });
    require(server.scheduler.snapshot().timed_out >= 1, "Deadline must release active work and increment metric");

    auto active = std::async(std::launch::async, [&] { return server.post(request("hold").dump()); });
    wait_for([&] { return server.scheduler.snapshot().active == 1; });
    check_response(server.post(body.dump()), 504);
    wait_for([&] { return server.scheduler.snapshot().queued == 0; });
    auto queued = std::async(std::launch::async, [&] { return server.post(request("hold").dump()); });
    wait_for([&] { return server.scheduler.snapshot().queued == 1; });
    server.scheduler.shutdown(false);
    check_response(active.get(), 503);
    check_response(queued.get(), 503);
    check_response(server.get("/health"), 503);
    check_response(server.post(request().dump()), 503);
    const auto metrics = check_response(server.get("/metrics"), 200);
    require(metrics.at("active") == 0 && metrics.at("queued") == 0 && !metrics.at("accepting").get<bool>(),
            "Shutdown must clear all active and queued work");
}
}  // namespace

int main() {
    try {
        test_base64();
        RunningServer server;
        auto invalid_options = options();
        invalid_options.http_threads = 3;
        must_throw([&] { HttpService invalid(server.scheduler, invalid_options, "synthetic"); });
        test_validation(server);
        test_queue_and_health(server);
        test_deadline_and_stop(server);
        server.service.stop();
        server.service.stop();
        require(!server.service.listen_after_bind(), "Stopped HTTP service must not restart");
        std::cout << "HTTP service validation passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
