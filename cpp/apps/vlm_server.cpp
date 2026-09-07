#include "qwen_vl/http_service.hpp"
#include "qwen_vl/service_backends.hpp"
#include "json.hpp"

#include <atomic>
#include <charconv>
#include <csignal>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <thread>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void interrupt_handler(int) { interrupted = 1; }

struct ListenerLifetime {
    qwen_vl::service::HttpService& server;
    qwen_vl::service::Scheduler& scheduler;
    std::thread& listener;

    void stop() {
        server.stop();
        scheduler.shutdown(false);
        if (listener.joinable()) listener.join();
    }

    ~ListenerLifetime() { stop(); }
};

int integer(const std::string& value) {
    int result = 0;
    const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
        throw std::invalid_argument("Invalid integer: " + value);
    return result;
}
}

int main(int argc, char** argv) {
    using namespace qwen_vl::service;
    try {
        std::map<std::string, std::string> flags;
        const std::map<std::string, bool> allowed = {
            {"--backend",true},{"--host",true},{"--port",true},{"--workers",true},
            {"--queue-capacity",true},{"--http-threads",true},{"--connection-queue",true},
            {"--timeout-ms",true},{"--max-timeout-ms",true},{"--max-tokens",true},
            {"--synthetic-delay-ms",true},{"--run-for-ms",true},
            {"--vision-engine",true},{"--vision-contract",true},{"--model",true},
            {"--ctx-size",true},{"--batch-size",true},{"--gpu-layers",true},{"--threads",true}
        };
        for (int i=1; i<argc; ++i) {
            const std::string flag=argv[i];
            if (flag=="--help") {
                std::cout << "vlm_server --backend synthetic|model [--host 127.0.0.1 --port 8080]\n"
                             "  --workers 1 --queue-capacity 8 --http-threads 16 --connection-queue 32\n"
                             "  --timeout-ms 30000 --max-timeout-ms 120000 --max-tokens 512\n"
                             "  --synthetic-delay-ms 25 --run-for-ms 0\n"
                             "Model: --vision-engine FILE --vision-contract FILE --model GGUF\n"
                             "  --ctx-size 4096 --batch-size 2048 --gpu-layers 99 --threads 0\n"
                             "GET /health, GET /metrics, POST /v1/generate\n";
                return 0;
            }
            if (!allowed.count(flag) || ++i==argc || flags.count(flag))
                throw std::invalid_argument("Unknown, missing or repeated argument: " + flag);
            flags[flag]=argv[i];
        }
        const auto number = [&](const std::string& key,int fallback) {
            return flags.count(key) ? integer(flags.at(key)) : fallback;
        };
        const auto text = [&](const std::string& key,const std::string& fallback) {
            return flags.count(key) ? flags.at(key) : fallback;
        };
        const auto backend_name=text("--backend", "");
        if (backend_name!="synthetic" && backend_name!="model")
            throw std::invalid_argument("Select --backend synthetic or --backend model explicitly");
        const int workers=number("--workers",1);
        const int capacity=number("--queue-capacity",8);
        const int http_threads=number("--http-threads",16);
        const int connections=number("--connection-queue",32);
        const int run_for_ms=number("--run-for-ms",0);
        if (workers<1 || workers>32 || capacity<1 || capacity>1024 ||
            http_threads<1 || http_threads>256 || connections<1 || connections>1024 ||
            run_for_ms<0 || run_for_ms>86400000)
            throw std::invalid_argument("Worker/queue/thread/duration limit out of range");
        if (backend_name=="model" && workers!=1)
            throw std::invalid_argument("Model backend uses one serialized inference worker");
        HttpOptions options;
        options.host=text("--host","127.0.0.1");
        options.port=number("--port",8080);
        options.workers=static_cast<std::size_t>(workers);
        options.queue_capacity=static_cast<std::size_t>(capacity);
        options.http_threads=static_cast<std::size_t>(http_threads);
        options.connection_queue=static_cast<std::size_t>(connections);
        options.default_timeout_ms=number("--timeout-ms",30000);
        options.max_timeout_ms=number("--max-timeout-ms",120000);
        options.max_tokens=number("--max-tokens",512);
        Backend backend;
        if (backend_name=="synthetic") {
            backend=make_synthetic_backend(std::chrono::milliseconds(number("--synthetic-delay-ms",25)));
        } else {
#if defined(QWEN_WITH_MODEL_BACKEND)
            ModelBackendOptions model_options;
            model_options.vision_engine=text("--vision-engine","");
            model_options.vision_contract=text("--vision-contract","");
            model_options.llama.model_path=text("--model","");
            model_options.llama.n_ctx=number("--ctx-size",4096);
            model_options.llama.n_batch=number("--batch-size",2048);
            model_options.llama.n_gpu_layers=number("--gpu-layers",99);
            model_options.llama.n_threads=number("--threads",0);
            model_options.llama.max_new_tokens=options.max_tokens;
            backend=make_model_backend(model_options);
#else
            throw std::invalid_argument("This executable requires QWEN_BUILD_INFERENCE=ON for the model backend");
#endif
        }
        Scheduler scheduler(static_cast<std::size_t>(workers),static_cast<std::size_t>(capacity),std::move(backend));
        HttpService server(scheduler,options,backend_name=="model" ? "tensorrt_vit_llamacpp" : "synthetic");
        int actual_port=options.port;
        if (actual_port==0) {
            actual_port=server.bind_to_any_port();
            if (actual_port<=0) throw std::runtime_error("Cannot bind an HTTP port");
        }
        std::signal(SIGINT,interrupt_handler);
        std::signal(SIGTERM,interrupt_handler);
        std::atomic<bool> done{false};
        std::atomic<bool> listen_ok{false};
        std::thread listener([&] {
            try { listen_ok.store(options.port==0 ? server.listen_after_bind() : server.listen()); }
            catch (...) { listen_ok.store(false); }
            done.store(true);
        });
        ListenerLifetime listener_lifetime{server, scheduler, listener};
        const auto start=Clock::now();
        while (!server.is_running() && !done.load() && Clock::now()-start<std::chrono::seconds(5))
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        if (!server.is_running()) {
            throw std::runtime_error("HTTP server startup failed");
        }
        std::cout << "LISTENING_JSON " << nlohmann::json({{"host",options.host},{"port",actual_port},
            {"backend",backend_name},{"workers",workers},{"queue_capacity",capacity}}).dump() << std::endl;
        while (!interrupted && !done.load()) {
            if (run_for_ms>0 && Clock::now()-start>=std::chrono::milliseconds(run_for_ms)) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        listener_lifetime.stop();
        return listen_ok.load() ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
