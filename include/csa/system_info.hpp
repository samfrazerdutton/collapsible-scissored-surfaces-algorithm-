// csa/system_info.hpp — a real, honestly-partial hardware/software
// fingerprint (`scissorc system`, brief section 200): every existing
// benchmark doc in this repository (REAL_GEO_BENCHMARK.md,
// docs/PARALLELISM.md, GPU_BENCHMARKS.md, ...) states its hardware by
// hand, in prose, once per document -- this is the first place that
// information is gathered by code instead of typed by hand, so a future
// benchmark run can attach it automatically instead of risking it going
// stale relative to the machine that actually produced the numbers.
//
// Every field that can't be determined portably reports itself as
// unavailable (an empty string / 0, surfaced by the CLI as "NOT
// AVAILABLE") rather than a guess -- matching this project's own
// AVAILABLE/UNAVAILABLE/NOT BUILT convention (see cuda_is_available()),
// not silently omitting the field.
#pragma once
#include "csa/common.hpp"
#include <string>

namespace csa {

struct SystemInfo {
    std::string os;              // compile-time: "Windows", "Linux", "macOS", or "Unknown"
    std::string compiler;        // e.g. "MSVC 19.29" or "GCC 13.3.0" or "Clang 18.1.3"
    std::string build_type;      // "Release" or "Debug", from whether NDEBUG is defined
    unsigned logical_cores = 0;  // std::thread::hardware_concurrency(); 0 if the platform can't report it
    std::string cpu_brand;       // x86 CPUID brand string; empty on non-x86 or if unavailable
    std::string simd_backend;    // "AVX2" or "Scalar", from detect_simd_backend()
    u64 ram_total_bytes = 0;     // 0 if unavailable on this platform
    bool cuda_available = false;
    std::string cuda_device_name;    // empty if cuda_available is false
    u64 cuda_device_memory_bytes = 0; // 0 if cuda_available is false
    std::string git_commit;      // short commit hash at configure time, or "unknown"
    bool git_dirty = false;      // true if the working tree had uncommitted changes at configure time
};

SystemInfo query_system_info();

} // namespace csa
