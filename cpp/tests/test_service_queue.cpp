#include "qwen_vl/service.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using namespace qwen_vl::service;
using namespace std::chrono_literals;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Request request(std::string id, Clock::time_point deadline = Clock::time_point::max()) {
    Request value;
    value.id = std::move(id);
    value.prompt = "test";
    value.deadline = deadline;
    return value;
}

Response get(const Submission& submission) {
    require(submission.future.valid(), "Missing response future");
    require(submission.future.wait_for(2s) == std::future_status::ready, "Response did not complete");
    return submission.future.get();
}

struct Gate {
    std::mutex mutex;
    std::condition_variable changed;
    unsigned entered = 0;
    bool released = false;

    Response run(const Request& input, const StopRequested& stop, bool cooperative = true) {
        std::unique_lock<std::mutex> lock(mutex);
        ++entered;
        changed.notify_all();
        while (!released && (!cooperative || !stop())) changed.wait_for(lock, 1ms);
        Response result;
        result.text = input.id;
        result.backend = "test";
        result.first_token_ms = 0.0;
        result.generated_tokens = 1;
        return result;
    }

    void wait(unsigned count) {
        std::unique_lock<std::mutex> lock(mutex);
        require(changed.wait_for(lock, 2s, [&] { return entered >= count; }), "Backend did not enter");
    }

    void release() {
        std::lock_guard<std::mutex> lock(mutex);
        released = true;
        changed.notify_all();
    }
};

struct UnblockOnExit {
    Gate& gate;
    ~UnblockOnExit() { gate.release(); }
};

void test_queue_full_and_cancel() {
    Gate gate;
    Scheduler scheduler(1, 1, [&](const Request& input, const StopRequested& stop) { return gate.run(input, stop); });
    auto active = scheduler.submit(request("active"));
    gate.wait(1);
    auto queued = scheduler.submit(request("queued"));
    require(scheduler.snapshot().queued == 1, "Pending queue size is incorrect");
    auto rejected = scheduler.submit(request("full"));
    require(rejected.status == Status::queue_full, "Full queue was not rejected");
    require(get(rejected).status == Status::queue_full, "Rejected future has wrong status");
    queued.cancel();
    queued.cancel();
    require(get(queued).status == Status::cancelled, "Queued cancellation failed");
    require(scheduler.snapshot().queued == 0, "Cancelled request retained its queue slot");
    auto replacement = scheduler.submit(request("replacement"));
    require(replacement.status == Status::ok, "Cancelled slot was not reusable");
    gate.release();
    require(get(active).text == "active", "Active response corrupted");
    const auto response = get(replacement);
    require(response.text == "replacement" && response.first_token_ms >= response.queue_ms, "Replacement timing or text is incorrect");
    scheduler.shutdown();
    auto stats = scheduler.snapshot();
    require(stats.accepted == 3 && stats.finished == 3 && stats.rejected == 1 && stats.cancelled == 1,
        "Queue/cancellation counters are incorrect");
    require(stats.active == 0 && stats.queued == 0 && stats.peak_active == 1, "Worker accounting is incorrect");
}

void test_concurrency_and_fifo() {
    Gate gate;
    Scheduler scheduler(2, 8, [&](const Request& input, const StopRequested& stop) { return gate.run(input, stop); });
    std::vector<Submission> submissions;
    submissions.push_back(scheduler.submit(request("one")));
    gate.wait(1);
    submissions.push_back(scheduler.submit(request("two")));
    gate.wait(2);
    for (int i = 0; i < 4; ++i) submissions.push_back(scheduler.submit(request(std::to_string(i))));
    const auto blocked = scheduler.snapshot();
    require(blocked.active == 2 && blocked.queued == 4 && blocked.peak_active == 2, "Concurrency bound violated");
    gate.release();
    scheduler.shutdown(true);
    for (const auto& submission : submissions) require(get(submission).status == Status::ok, "Drain lost work");
    require(scheduler.snapshot().finished == 6 && scheduler.snapshot().peak_active == 2, "Drain counters are incorrect");

    Gate serial_gate;
    std::vector<std::string> order;
    Scheduler serial(1, 8, [&](const Request& input, const StopRequested& stop) {
        order.push_back(input.id);
        return serial_gate.run(input, stop);
    });
    auto first = serial.submit(request("first"));
    serial_gate.wait(1);
    auto second = serial.submit(request("second"));
    auto third = serial.submit(request("third"));
    serial_gate.release();
    serial.shutdown(true);
    require(order == std::vector<std::string>({"first", "second", "third"}), "Queue is not FIFO");
    require(get(first).status == Status::ok && get(second).status == Status::ok && get(third).status == Status::ok,
        "FIFO response failed");
}

void test_deadlines() {
    Gate gate;
    Scheduler scheduler(1, 1, [&](const Request& input, const StopRequested& stop) { return gate.run(input, stop); });
    auto active = scheduler.submit(request("active"));
    gate.wait(1);
    auto queued = scheduler.submit(request("deadline", Clock::now() + 25ms));
    require(get(queued).status == Status::timeout, "Queued deadline was not enforced while worker blocked");
    require(scheduler.snapshot().queued == 0, "Timed-out request retained its queue slot");
    auto replacement = scheduler.submit(request("replacement"));
    require(replacement.status == Status::ok, "Expired slot was not reusable");
    auto expired = scheduler.submit(request("expired", Clock::now() - 1ms));
    require(get(expired).status == Status::timeout, "Already expired request was not completed");
    gate.release();
    scheduler.shutdown(true);
    require(get(active).status == Status::ok && get(replacement).status == Status::ok, "Deadline affected unrelated work");
    require(gate.entered == 2 && scheduler.snapshot().timed_out == 2, "Expired backend executed or timeout count is wrong");

    Gate active_gate;
    Scheduler active_scheduler(1, 2, [&](const Request& input, const StopRequested& stop) { return active_gate.run(input, stop); });
    auto timed = active_scheduler.submit(request("active_deadline", Clock::now() + 50ms));
    active_gate.wait(1);
    require(get(timed).status == Status::timeout, "Active deadline was not enforced");
    active_scheduler.shutdown(true);
    require(active_scheduler.snapshot().finished == 1 && active_scheduler.snapshot().active == 0, "Active timeout completed twice");
}

void test_active_cancellation() {
    Gate gate;
    Scheduler scheduler(1, 2, [&](const Request& input, const StopRequested& stop) { return gate.run(input, stop); });
    auto active = scheduler.submit(request("active"));
    gate.wait(1);
    active.cancel();
    require(get(active).status == Status::cancelled, "Active cancellation failed");
    scheduler.shutdown();
    active.cancel();
    require(scheduler.snapshot().finished == 1 && scheduler.snapshot().cancelled == 1, "Repeated cancel completed twice");
}

void test_timeout_retains_worker_slot() {
    Gate gate;
    Scheduler scheduler(1, 1, [&](const Request& input, const StopRequested& stop) {
        return gate.run(input, stop, false);
    });
    UnblockOnExit unblock{gate};
    auto timed = scheduler.submit(request("non_interruptible", Clock::now() + 50ms));
    gate.wait(1);
    require(get(timed).status == Status::timeout, "Uninterruptible backend blocked timeout response");
    require(scheduler.snapshot().active == 1, "Timeout prematurely released an executing worker");
    auto queued = scheduler.submit(request("queued"));
    require(scheduler.snapshot().queued == 1, "Queued request bypassed concurrency limit");
    gate.release();
    scheduler.shutdown(true);
    require(get(queued).status == Status::ok, "Worker did not resume after timed-out backend returned");
    require(scheduler.snapshot().peak_active == 1 && scheduler.snapshot().finished == 2, "Late backend result completed twice");
}

void test_exceptions_and_shutdown() {
    Scheduler scheduler(1, 4, [](const Request& input, const StopRequested&) {
        if (input.id == "throw") throw std::runtime_error("backend failure");
        if (input.id == "unknown") throw 7;
        Response result;
        if (input.id == "invalid") result.status = Status::invalid_request;
        return result;
    });
    auto throwing = scheduler.submit(request("throw"));
    require(get(throwing).status == Status::backend_error && get(throwing).error == "backend failure", "Exception was not isolated");
    require(get(scheduler.submit(request("unknown"))).status == Status::backend_error, "Nonstandard exception killed worker");
    require(get(scheduler.submit(request("invalid"))).status == Status::invalid_request, "Invalid status was lost");
    require(get(scheduler.submit(request("ok"))).status == Status::ok, "Worker did not recover from exception");
    scheduler.shutdown();
    scheduler.shutdown(false);
    auto stopped = scheduler.submit(request("stopped"));
    require(stopped.status == Status::stopping && get(stopped).status == Status::stopping, "Stopped scheduler accepted work");
    require(scheduler.snapshot().failed == 3 && scheduler.snapshot().finished == 4 && !scheduler.snapshot().accepting,
        "Failure/shutdown counters are incorrect");

    Gate gate;
    Scheduler cancelling(1, 2, [&](const Request& input, const StopRequested& stop) { return gate.run(input, stop); });
    auto active = cancelling.submit(request("active"));
    gate.wait(1);
    auto queued = cancelling.submit(request("queued"));
    cancelling.shutdown(false);
    require(get(active).status == Status::cancelled && get(queued).status == Status::cancelled, "Shutdown did not cancel work");
    require(gate.entered == 1 && cancelling.snapshot().active == 0, "Shutdown ran queued work or did not join workers");

    Scheduler* self = nullptr;
    Scheduler reentrant(1, 1, [&](const Request&, const StopRequested&) {
        self->shutdown(false);
        return Response{};
    });
    self = &reentrant;
    auto response = reentrant.submit(request("self_shutdown"));
    require(get(response).status == Status::cancelled, "Worker shutdown deadlocked");
    reentrant.shutdown();
}

void test_cancel_lifetime_and_validation() {
    std::function<void()> cancel;
    {
        Scheduler scheduler(1, 1, [](const Request&, const StopRequested&) { return Response{}; });
        auto result = scheduler.submit(request("lifetime"));
        cancel = result.cancel;
        require(get(result).status == Status::ok, "Lifetime request failed");
    }
    cancel();
    const Backend backend = [](const Request&, const StopRequested&) { return Response{}; };
    for (const auto parameters : {std::pair<unsigned, unsigned>{0, 1}, {1, 0}}) {
        bool threw = false;
        try { Scheduler invalid(parameters.first, parameters.second, backend); }
        catch (const std::invalid_argument&) { threw = true; }
        require(threw, "Invalid scheduler bounds were accepted");
    }
    bool threw = false;
    try { Scheduler invalid(1, 1, {}); }
    catch (const std::invalid_argument&) { threw = true; }
    require(threw, "Missing backend was accepted");
}
}  // namespace

int main() {
    try {
        test_queue_full_and_cancel();
        test_concurrency_and_fifo();
        test_deadlines();
        test_active_cancellation();
        test_timeout_retains_worker_slot();
        test_exceptions_and_shutdown();
        test_cancel_lifetime_and_validation();
        std::cout << "Service queue checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
