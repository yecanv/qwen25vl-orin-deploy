#include "qwen_vl/service_backends.hpp"

#include <algorithm>
#include <stdexcept>
#include <thread>

namespace qwen_vl::service {
Backend make_synthetic_backend(std::chrono::milliseconds delay) {
    if (delay.count() < 0 || delay > std::chrono::minutes(5))
        throw std::invalid_argument("synthetic delay must be 0..300000 ms");
    return [delay](const Request& request, const StopRequested& stop) {
        Response response;
        response.id = request.id;
        response.backend = "synthetic";
        if (request.max_new_tokens < 0) {
            response.status = Status::invalid_request;
            response.error = "max_new_tokens must be nonnegative";
            return response;
        }
        const auto start = Clock::now();
        const auto until = start + delay;
        while (Clock::now() < until) {
            if (stop()) {
                response.status = Status::cancelled;
                response.error = "Request cancelled";
                return response;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        if (stop()) {
            response.status = Status::cancelled;
            response.error = "Request cancelled";
            return response;
        }
        response.generated_tokens = std::min(request.max_new_tokens, 8);
        if (response.generated_tokens > 0) {
            response.text = "synthetic: " + request.prompt;
            response.first_token_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        }
        return response;
    };
}
}  // namespace qwen_vl::service
