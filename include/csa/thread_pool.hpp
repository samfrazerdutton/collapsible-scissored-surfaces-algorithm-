// csa/thread_pool.hpp — a real, reusable worker-thread pool, built to
// replace the "spawn N std::thread objects, join them, throw them away"
// pattern that src/rans_coder.cpp's run_lanes() used until this file
// existed. That pattern is correct but wasteful under repeated use (a
// long-running process compressing many files pays full OS thread
// creation/teardown cost on every single call); this pool creates its
// worker threads once and reuses them for every submitted task for the
// pool's lifetime.
//
// This is intentionally a small, general-purpose primitive (a bounded
// number of long-lived workers pulling from one shared FIFO queue), not
// a work-stealing scheduler: the actual workload this project has today
// (encode/decode one lane per worker, a handful of roughly-equal-sized
// independent tasks per call) has no per-task heterogeneity worth
// stealing across, so a work-stealing queue would add real complexity
// for no measured benefit -- see docs/PARALLELISM.md for the measurement
// that motivated stopping here rather than building more.
#pragma once
#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace csa {

class ThreadPool {
public:
    // 0 or an unspecified count falls back to hardware_concurrency(),
    // itself clamped to at least 1 (hardware_concurrency() is allowed by
    // the standard to return 0 when it cannot be determined).
    explicit ThreadPool(size_t num_threads = 0) {
        if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; i++) workers_.emplace_back([this] { worker_loop(); });
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Graceful shutdown: signals every worker to stop once its current
    // task (and anything already queued) finishes, then joins all of
    // them. No thread is ever detached or leaked.
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) t.join();
    }

    size_t size() const { return workers_.size(); }

    // Submits fn to run on a worker thread; returns a future that carries
    // fn's return value or, if fn throws, the exception (std::packaged_task
    // captures it automatically, so a worker thread throwing can never
    // bring down the whole process -- the exception surfaces at the
    // future's .get() call instead, on whichever thread is waiting on it).
    template <typename Fn, typename R = std::invoke_result_t<Fn>>
    std::future<R> submit(Fn&& fn) {
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
        std::future<R> fut = task->get_future();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stop_) throw std::runtime_error("csa::ThreadPool: submit() called after shutdown");
            tasks_.emplace([task] { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                if (tasks_.empty()) {
                    if (stop_) return;
                    continue;
                }
                task = std::move(tasks_.front());
                tasks_.pop();
            }
            task();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stop_ = false;
};

// Process-wide default pool, sized to hardware_concurrency() once on
// first use (thread-safe lazy init via a C++11 function-local static --
// no separate initialization step callers need to remember to run).
// Every real caller in this codebase today (run_lanes(), see
// rans_coder.cpp) goes through this single shared instance rather than
// building its own pool per call, so worker threads are created once for
// the process's lifetime, not once per compress()/decompress() call.
inline ThreadPool& default_thread_pool() {
    static ThreadPool pool;
    return pool;
}

// Runs fn(0..count-1), one call per index, using the default pool, and
// waits for every call to finish before returning -- the same fn(i)
// contract src/rans_coder.cpp's old run_lanes() had, so it's a drop-in
// replacement. Exceptions thrown by any fn(i) propagate out of this call
// (the first one encountered while collecting futures; the others are
// still waited on so no task outlives this call).
template <typename Fn>
void parallel_for(int count, Fn&& fn) {
    if (count <= 0) return;
    if (count == 1) { fn(0); return; }
    ThreadPool& pool = default_thread_pool();
    std::vector<std::future<void>> futures;
    futures.reserve((size_t)count);
    for (int i = 0; i < count; i++) futures.push_back(pool.submit([&fn, i] { fn(i); }));
    std::exception_ptr first_error;
    for (auto& f : futures) {
        try {
            f.get();
        } catch (...) {
            if (!first_error) first_error = std::current_exception();
        }
    }
    if (first_error) std::rethrow_exception(first_error);
}

} // namespace csa
