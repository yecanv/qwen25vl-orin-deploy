#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <vector>

namespace qwen_vl::service {

using Clock = std::chrono::steady_clock;
enum class Status { ok, queue_full, timeout, cancelled, stopping, backend_error, invalid_request };

struct Request {
    std::string id;
    std::string prompt;
    std::vector<std::uint8_t> image;
    int max_new_tokens = 128;
    Clock::time_point deadline = Clock::time_point::max();
};

struct Response {
    Status status = Status::ok;
    std::string id;
    std::string text;
    std::string error;
    std::string backend;
    int generated_tokens = 0;
    double queue_ms = 0.0;
    double inference_ms = 0.0;
    double first_token_ms = -1.0;
};

using StopRequested = std::function<bool()>;
// Backend first_token_ms starts at backend entry; Scheduler adds queue_ms.
// Running cancellation is cooperative and holds its worker until Backend returns.
using Backend = std::function<Response(const Request&, const StopRequested&)>;

struct Submission {
    Status status = Status::ok;
    std::shared_future<Response> future;
    std::function<void()> cancel;
};

struct Snapshot {
    std::uint64_t accepted = 0;
    std::uint64_t finished = 0;
    std::uint64_t rejected = 0;
    std::uint64_t timed_out = 0;
    std::uint64_t cancelled = 0;
    std::uint64_t failed = 0;
    std::size_t queued = 0;
    std::size_t active = 0;
    std::size_t peak_active = 0;
    bool accepting = true;
};

class Scheduler {
public:
    Scheduler(std::size_t workers, std::size_t queue_capacity, Backend backend);
    ~Scheduler();
    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;
    Scheduler(Scheduler&&) = delete;
    Scheduler& operator=(Scheduler&&) = delete;

    Submission submit(Request request);
    Snapshot snapshot() const;
    void shutdown(bool drain = true);

private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

}  // namespace qwen_vl::service
