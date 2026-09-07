#include "qwen_vl/service.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <exception>
#include <list>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

namespace qwen_vl::service {
namespace {
thread_local const void* scheduler_thread = nullptr;

double milliseconds(Clock::duration duration) {
    return std::chrono::duration<double, std::milli>(duration).count();
}

Response error_response(Status status, const std::string& id, const std::string& error) {
    Response result;
    result.status = status;
    result.id = id;
    result.error = error;
    return result;
}
}  // namespace

struct Scheduler::Impl : std::enable_shared_from_this<Scheduler::Impl> {
    struct Job {
        explicit Job(Request value) : request(std::move(value)), queued_at(Clock::now()) {}
        Request request;
        const Clock::time_point queued_at;
        Clock::time_point started_at{};
        std::promise<Response> promise;
        std::atomic<bool> stop{false};
        bool started = false;
        bool done = false;
    };

    Impl(std::size_t capacity, Backend callback)
        : queue_capacity(capacity), backend(std::move(callback)) {}

    ~Impl() {
        // A backend may destroy its owner; thread captures retain this state until exit.
        for (auto& worker : workers) if (worker.joinable()) worker.detach();
        if (timer.joinable()) timer.detach();
    }

    void start(std::size_t count) {
        workers.reserve(count);
        try {
            for (std::size_t i = 0; i < count; ++i) {
                workers.emplace_back([self = shared_from_this()] { self->work(); });
            }
            timer = std::thread([self = shared_from_this()] { self->monitor(); });
        } catch (...) {
            shutdown(false);
            throw;
        }
    }

    void finish(const std::shared_ptr<Job>& job, Response response, Clock::time_point now) {
        if (job->done) return;
        response.id = job->request.id;
        response.queue_ms = milliseconds((job->started ? job->started_at : now) - job->queued_at);
        response.inference_ms = job->started ? milliseconds(now - job->started_at) : 0.0;
        if (response.first_token_ms >= 0.0) response.first_token_ms += response.queue_ms;
        switch (response.status) {
        case Status::timeout: ++stats.timed_out; break;
        case Status::cancelled: ++stats.cancelled; break;
        case Status::backend_error:
        case Status::invalid_request: ++stats.failed; break;
        default: break;
        }
        job->done = true;
        job->stop.store(true, std::memory_order_release);
        ++stats.finished;
        job->promise.set_value(std::move(response));
    }

    void expire(Clock::time_point now) {
        for (auto it = queue.begin(); it != queue.end();) {
            const auto& job = *it;
            if (job->request.deadline <= now) {
                finish(job, error_response(Status::timeout, job->request.id, "Request deadline exceeded"), now);
                it = queue.erase(it);
            } else {
                ++it;
            }
        }
        for (const auto& job : running) {
            if (!job->done && job->request.deadline <= now) {
                finish(job, error_response(Status::timeout, job->request.id, "Request deadline exceeded"), now);
            }
        }
    }

    Submission submit(Request request) {
        auto job = std::make_shared<Job>(std::move(request));
        Submission result;
        result.future = job->promise.get_future().share();
        result.cancel = [] {};
        {
            std::lock_guard<std::mutex> lock(mutex);
            const auto now = Clock::now();
            expire(now);
            if (!stats.accepting) {
                result.status = Status::stopping;
            } else if (job->request.deadline > now && queue.size() >= queue_capacity) {
                result.status = Status::queue_full;
            }
            if (result.status != Status::ok) {
                ++stats.rejected;
                job->promise.set_value(error_response(result.status, job->request.id,
                    result.status == Status::stopping ? "Scheduler is stopping" : "Request queue is full"));
                return result;
            }
            ++stats.accepted;
            if (job->request.deadline <= now) {
                finish(job, error_response(Status::timeout, job->request.id, "Request deadline exceeded"), now);
            } else {
                queue.push_back(job);
            }
            result.cancel = [owner = weak_from_this(), pending = std::weak_ptr<Job>(job)] {
                if (const auto self = owner.lock()) {
                    if (const auto item = pending.lock()) self->cancel(item);
                }
            };
        }
        changed.notify_all();
        return result;
    }

    void cancel(const std::shared_ptr<Job>& job) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            if (job->done) return;
            const auto now = Clock::now();
            const bool expired = job->request.deadline <= now;
            finish(job, error_response(expired ? Status::timeout : Status::cancelled,
                job->request.id, expired ? "Request deadline exceeded" : "Request cancelled"), now);
            if (!job->started) queue.remove(job);
        }
        changed.notify_all();
    }

    void work() {
        scheduler_thread = this;
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] { return !stats.accepting || !queue.empty(); });
                expire(Clock::now());
                if (queue.empty()) {
                    if (!stats.accepting) break;
                    continue;
                }
                job = queue.front();
                queue.pop_front();
                job->started = true;
                job->started_at = Clock::now();
                running.push_back(job);
                stats.peak_active = std::max(stats.peak_active, running.size());
            }
            changed.notify_all();
            Response response;
            try {
                response = backend(job->request, [job] {
                    return job->stop.load(std::memory_order_acquire) || Clock::now() >= job->request.deadline;
                });
            } catch (const std::exception& error) {
                response = error_response(Status::backend_error, job->request.id, error.what());
            } catch (...) {
                response = error_response(Status::backend_error, job->request.id, "Unknown backend exception");
            }
            {
                std::lock_guard<std::mutex> lock(mutex);
                const auto now = Clock::now();
                if (!job->done && job->request.deadline <= now) {
                    response = error_response(Status::timeout, job->request.id, "Request deadline exceeded");
                }
                finish(job, std::move(response), now);
                running.remove(job);
            }
            changed.notify_all();
        }
        scheduler_thread = nullptr;
    }

    void monitor() {
        scheduler_thread = this;
        std::unique_lock<std::mutex> lock(mutex);
        for (;;) {
            expire(Clock::now());
            if (!stats.accepting && queue.empty() && running.empty()) break;
            auto deadline = Clock::time_point::max();
            for (const auto& job : queue) deadline = std::min(deadline, job->request.deadline);
            for (const auto& job : running) {
                if (!job->done) deadline = std::min(deadline, job->request.deadline);
            }
            if (deadline == Clock::time_point::max()) changed.wait(lock);
            else changed.wait_until(lock, deadline);
        }
        scheduler_thread = nullptr;
    }

    Snapshot snapshot() const {
        std::lock_guard<std::mutex> lock(mutex);
        auto result = stats;
        result.queued = queue.size();
        result.active = running.size();
        return result;
    }

    void shutdown(bool drain) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            stats.accepting = false;
            if (!drain) {
                const auto now = Clock::now();
                for (const auto& job : queue) {
                    finish(job, error_response(Status::cancelled, job->request.id, "Scheduler stopped"), now);
                }
                queue.clear();
                for (const auto& job : running) {
                    finish(job, error_response(Status::cancelled, job->request.id, "Scheduler stopped"), now);
                }
            }
        }
        changed.notify_all();
        // A worker can request shutdown, but cannot synchronously join itself.
        if (scheduler_thread == this) return;
        std::lock_guard<std::mutex> join_lock(join_mutex);
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        if (timer.joinable()) timer.join();
    }

    const std::size_t queue_capacity;
    const Backend backend;
    mutable std::mutex mutex;
    std::mutex join_mutex;
    std::condition_variable changed;
    std::list<std::shared_ptr<Job>> queue;
    std::list<std::shared_ptr<Job>> running;
    Snapshot stats;
    std::vector<std::thread> workers;
    std::thread timer;
};

Scheduler::Scheduler(std::size_t workers, std::size_t queue_capacity, Backend backend) {
    if (workers == 0 || queue_capacity == 0 || !backend) {
        throw std::invalid_argument("Scheduler requires workers, queue capacity and a backend");
    }
    impl_ = std::make_shared<Impl>(queue_capacity, std::move(backend));
    impl_->start(workers);
}

Scheduler::~Scheduler() { impl_->shutdown(false); }

Submission Scheduler::submit(Request request) { return impl_->submit(std::move(request)); }

Snapshot Scheduler::snapshot() const { return impl_->snapshot(); }

void Scheduler::shutdown(bool drain) { impl_->shutdown(drain); }

}  // namespace qwen_vl::service
