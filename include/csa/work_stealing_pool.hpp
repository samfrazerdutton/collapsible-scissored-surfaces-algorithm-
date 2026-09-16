// csa/work_stealing_pool.hpp — a real work-stealing thread pool, built
// and kept as a separate primitive from csa::ThreadPool (thread_pool.hpp)
// rather than replacing it: ThreadPool's one real caller in this
// codebase (the interleaved rANS lanes, src/rans_coder.cpp) submits a
// handful of same-sized independent tasks per call, which a plain
// shared-queue pool already handles with no measured load imbalance --
// see docs/PARALLELISM.md's own reasoning for not building a
// work-stealing scheduler speculatively for that workload. This class
// exists for the opposite, genuinely real case: a batch of tasks whose
// individual costs vary a lot and aren't known in advance (e.g.
// compressing many files of very different sizes, or running
// scissorc optimize's per-candidate search across several files at
// once) -- exactly the shape of workload a single shared FIFO queue
// handles fine (any idle worker just pulls the next task) but a naive
// *static* round-robin split across N independent per-worker queues
// does not: whichever worker draws the expensive tasks finishes last
// while the others sit idle with empty queues. See
// tests/test_main.cpp's test_work_stealing_pool for a real, measured
// case built specifically to exhibit that imbalance and confirm
// stealing actually corrects it, not just assert the class compiles.
//
// Implementation: each worker owns one double-ended queue of pending
// tasks, guarded by its own mutex (deliberately not a lock-free
// Chase-Lev deque -- that data structure is notoriously easy to get
// subtly wrong under concurrent steal/pop races, and this codebase does
// not yet have the extensive stress-testing such a structure deserves
// before being trusted; a per-worker mutex is slower under heavy
// contention but its correctness is straightforward to reason about and
// to verify under ThreadSanitizer -- see docs/PARALLELISM.md). A worker
// pops from the *front* of its own queue (cheap, no contention with
// steals) and, when its own queue is empty, scans other workers' queues
// and steals from the *back* (the oldest, usually largest-grained
// remaining task, and the end least likely to be contended by the
// owning worker's own front-pops).
#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace csa {

// Real, measured scheduler statistics -- see WorkStealingPool::stats().
// Every field here is a live counter incremented at the exact point the
// event happens, never a value computed after the fact or estimated.
struct WorkStealingStats {
    size_t tasks_submitted = 0;
    size_t tasks_completed = 0;
    size_t steal_attempts = 0;
    size_t steals_succeeded = 0;
};

class WorkStealingPool {
public:
    explicit WorkStealingPool(size_t num_threads = 0) {
        if (num_threads == 0) num_threads = std::thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 1;
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; i++) workers_.push_back(std::make_unique<WorkerState>());
        threads_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; i++) threads_.emplace_back([this, i] { worker_loop(i); });
    }

    WorkStealingPool(const WorkStealingPool&) = delete;
    WorkStealingPool& operator=(const WorkStealingPool&) = delete;

    ~WorkStealingPool() {
        stop_.store(true, std::memory_order_relaxed);
        cv_.notify_all();
        for (auto& t : threads_) t.join();
    }

    size_t size() const { return workers_.size(); }

    // Round-robins external submissions across worker queues (this
    // function is normally called from outside any worker thread, so
    // there is no "local" queue to prefer the way a worker stealing for
    // itself would). tasks_submitted() is incremented here, at the real
    // moment of submission, not inferred later.
    template <typename Fn, typename R = std::invoke_result_t<Fn>>
    std::future<R> submit(Fn&& fn) {
        auto task = std::make_shared<std::packaged_task<R()>>(std::forward<Fn>(fn));
        std::future<R> fut = task->get_future();
        size_t target = next_queue_.fetch_add(1, std::memory_order_relaxed) % workers_.size();
        {
            std::lock_guard<std::mutex> lock(workers_[target]->mutex);
            workers_[target]->queue.push_back([task] { (*task)(); });
        }
        stats_.tasks_submitted.fetch_add(1, std::memory_order_relaxed);
        cv_.notify_all();
        return fut;
    }

    WorkStealingStats stats() const {
        return {
            stats_.tasks_submitted.load(std::memory_order_relaxed),
            stats_.tasks_completed.load(std::memory_order_relaxed),
            stats_.steal_attempts.load(std::memory_order_relaxed),
            stats_.steals_succeeded.load(std::memory_order_relaxed),
        };
    }

private:
    struct WorkerState {
        std::mutex mutex;
        std::deque<std::function<void()>> queue;
    };
    struct AtomicStats {
        std::atomic<size_t> tasks_submitted{0};
        std::atomic<size_t> tasks_completed{0};
        std::atomic<size_t> steal_attempts{0};
        std::atomic<size_t> steals_succeeded{0};
    };

    bool try_pop_own(size_t idx, std::function<void()>& out) {
        std::lock_guard<std::mutex> lock(workers_[idx]->mutex);
        auto& q = workers_[idx]->queue;
        if (q.empty()) return false;
        out = std::move(q.front());
        q.pop_front();
        return true;
    }

    bool try_steal(size_t thief_idx, std::function<void()>& out) {
        for (size_t offset = 1; offset < workers_.size(); offset++) {
            size_t victim = (thief_idx + offset) % workers_.size();
            stats_.steal_attempts.fetch_add(1, std::memory_order_relaxed);
            std::lock_guard<std::mutex> lock(workers_[victim]->mutex);
            auto& q = workers_[victim]->queue;
            if (q.empty()) continue;
            out = std::move(q.back()); // steal from the opposite end from where the owner pops
            q.pop_back();
            stats_.steals_succeeded.fetch_add(1, std::memory_order_relaxed);
            return true;
        }
        return false;
    }

    void worker_loop(size_t idx) {
        for (;;) {
            std::function<void()> task;
            if (try_pop_own(idx, task) || try_steal(idx, task)) {
                task();
                stats_.tasks_completed.fetch_add(1, std::memory_order_relaxed);
                continue;
            }
            if (stop_.load(std::memory_order_relaxed)) return;
            std::unique_lock<std::mutex> lock(cv_mutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(1), [this] { return stop_.load(std::memory_order_relaxed); });
        }
    }

    std::vector<std::unique_ptr<WorkerState>> workers_;
    std::vector<std::thread> threads_;
    std::atomic<size_t> next_queue_{0};
    std::atomic<bool> stop_{false};
    std::mutex cv_mutex_;
    std::condition_variable cv_;
    AtomicStats stats_;
};

} // namespace csa
