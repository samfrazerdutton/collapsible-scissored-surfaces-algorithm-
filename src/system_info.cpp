#include "csa/system_info.hpp"
#include "csa/simd.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include <cstdio>
#include <thread>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#elif defined(__linux__)
#include <sys/sysinfo.h>
#endif

#if CSA_X86_SIMD
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#endif
#endif

namespace csa {

namespace {

std::string detect_os() {
#if defined(_WIN32)
    return "Windows";
#elif defined(__APPLE__)
    return "macOS";
#elif defined(__linux__)
    return "Linux";
#else
    return "Unknown";
#endif
}

std::string detect_compiler() {
    char buf[64];
#if defined(__clang__)
    std::snprintf(buf, sizeof(buf), "Clang %d.%d.%d", __clang_major__, __clang_minor__, __clang_patchlevel__);
#elif defined(_MSC_VER)
    std::snprintf(buf, sizeof(buf), "MSVC %d", _MSC_VER);
#elif defined(__GNUC__)
    std::snprintf(buf, sizeof(buf), "GCC %d.%d.%d", __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
#else
    return "Unknown";
#endif
    return std::string(buf);
}

std::string detect_cpu_brand() {
#if CSA_X86_SIMD
    int regs[4] = {0, 0, 0, 0};
#if defined(_MSC_VER)
    __cpuid(regs, 0x80000000);
#else
    __get_cpuid(0x80000000, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
#endif
    unsigned int max_ext_leaf = (unsigned int)regs[0];
    if (max_ext_leaf < 0x80000004) return "";

    char brand[49] = {0};
    for (int leaf = 0; leaf < 3; leaf++) {
#if defined(_MSC_VER)
        __cpuid(regs, 0x80000002 + leaf);
#else
        __get_cpuid(0x80000002 + leaf, (unsigned int*)&regs[0], (unsigned int*)&regs[1], (unsigned int*)&regs[2], (unsigned int*)&regs[3]);
#endif
        for (int r = 0; r < 4; r++)
            for (int b = 0; b < 4; b++)
                brand[leaf * 16 + r * 4 + b] = (char)((regs[r] >> (b * 8)) & 0xFF);
    }
    // Brand strings are often padded with spaces by the CPU itself, on
    // either end.
    std::string s(brand);
    size_t start = s.find_first_not_of(' ');
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(' ');
    return s.substr(start, end - start + 1);
#else
    return "";
#endif
}

u64 detect_ram_total_bytes() {
#if defined(_WIN32)
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status)) return (u64)status.ullTotalPhys;
    return 0;
#elif defined(__linux__)
    struct sysinfo info;
    if (sysinfo(&info) == 0) return (u64)info.totalram * (u64)info.mem_unit;
    return 0;
#else
    return 0;
#endif
}

} // namespace

SystemInfo query_system_info() {
    SystemInfo info;
    info.os = detect_os();
    info.compiler = detect_compiler();
#if defined(NDEBUG)
    info.build_type = "Release";
#else
    info.build_type = "Debug";
#endif
    info.logical_cores = std::thread::hardware_concurrency();
    info.cpu_brand = detect_cpu_brand();
    info.simd_backend = (detect_simd_backend() == SimdBackend::AVX2) ? "AVX2" : "Scalar";
    info.ram_total_bytes = detect_ram_total_bytes();
    info.cuda_available = cuda_device_info(info.cuda_device_name, info.cuda_device_memory_bytes);
    return info;
}

} // namespace csa
