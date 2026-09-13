// Exercises libcsa's C ABI (csa_capi.h) by linking against the actual
// shared library (csa_capi), not the C++ core directly -- this is the
// only way to actually catch a real symbol-export/linking problem (e.g.
// a missing __declspec(dllexport) on Windows), as opposed to just
// re-verifying the C++ logic that's already covered by csa_tests.
#include "csa/csa_capi.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        std::printf("FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
    } \
} while (0)

static void test_general() {
    std::vector<unsigned char> input;
    for (int i = 0; i < 500; i++) {
        const char* s = "the quick brown fox jumps over the lazy dog. ";
        for (const char* p = s; *p; p++) input.push_back((unsigned char)*p);
    }

    csa_buffer compressed = csa_compress(input.data(), input.size(), 0);
    CHECK(compressed.data != nullptr);
    CHECK(compressed.size > 0 && compressed.size < input.size() / 10);

    csa_buffer decompressed = csa_decompress(compressed.data, compressed.size);
    CHECK(decompressed.size == input.size());
    CHECK(decompressed.data != nullptr &&
          std::memcmp(decompressed.data, input.data(), input.size()) == 0);

    csa_free_buffer(compressed);
    csa_free_buffer(decompressed);
}

static void test_geo2d() {
    std::vector<int32_t> xy;
    double x = 50, y = 0, dx = 3, dy = 0;
    for (int i = 0; i < 300; i++) {
        xy.push_back((int32_t)(x + 0.5));
        xy.push_back((int32_t)(y + 0.5));
        double ndx = 1.02 * (dx * 0.9887 - dy * 0.1494); // fixed small rotation+scale
        double ndy = 1.02 * (dx * 0.1494 + dy * 0.9887);
        dx = ndx; dy = ndy;
        x += dx; y += dy;
    }
    size_t count = xy.size() / 2;

    csa_buffer blob = csa_compress_geo2d(xy.data(), count);
    CHECK(blob.data != nullptr && blob.size > 0);

    size_t out_count = 0;
    csa_buffer decoded = csa_decompress_geo2d(blob.data, blob.size, &out_count);
    CHECK(out_count == count);
    CHECK(decoded.data != nullptr);
    if (decoded.data) {
        const int32_t* out_xy = (const int32_t*)decoded.data;
        bool match = true;
        for (size_t i = 0; i < xy.size(); i++)
            if (out_xy[i] != xy[i]) match = false;
        CHECK(match);
    }

    csa_free_buffer(blob);
    csa_free_buffer(decoded);
}

static void test_geo2d_lossy() {
    std::vector<int32_t> xy;
    // A smooth-ish noisy path, not jagged modular-arithmetic noise --
    // lossy mode's whole premise is quantizing prediction residuals on
    // data the rotation+scale predictor can actually track.
    double x = 0, y = 0, heading = 0;
    unsigned int seed = 99;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    for (int i = 0; i < 2000; i++) {
        heading += (next_rand() - 0.5) * 0.1;
        double speed = 8.0 + (next_rand() - 0.5);
        x += speed * std::cos(heading);
        y += speed * std::sin(heading);
        xy.push_back((int32_t)(x + 0.5));
        xy.push_back((int32_t)(y + 0.5));
    }
    size_t count = xy.size() / 2;

    csa_buffer lossless = csa_compress_geo2d(xy.data(), count);
    csa_buffer lossy = csa_compress_geo2d_lossy(xy.data(), count, /*quant_step=*/25, /*resync_interval=*/32);
    CHECK(lossy.size > 0);
    CHECK(lossy.size < lossless.size);

    size_t out_count = 0;
    csa_buffer decoded = csa_decompress_geo2d(lossy.data, lossy.size, &out_count);
    CHECK(out_count == count);
    // Bounded error, not exact equality (that's the point of lossy mode).
    if (decoded.data) {
        const int32_t* out_xy = (const int32_t*)decoded.data;
        int32_t max_err = 0;
        for (size_t i = 0; i < xy.size(); i++) {
            int32_t d = out_xy[i] - xy[i];
            if (d < 0) d = -d;
            if (d > max_err) max_err = d;
        }
        CHECK(max_err < 25 * 40); // generous bound, same spirit as csa_tests' check
    }

    csa_free_buffer(lossless);
    csa_free_buffer(lossy);
    csa_free_buffer(decoded);
}

static void test_error_reporting() {
    unsigned char garbage[] = {1, 2, 3, 4, 5};
    csa_buffer result = csa_decompress(garbage, sizeof(garbage));
    CHECK(result.data == nullptr);
    CHECK(result.size == 0);
    const char* err = csa_last_error();
    CHECK(err != nullptr && std::strlen(err) > 0);
    csa_free_buffer(result);
}

int main() {
    test_general();
    test_geo2d();
    test_geo2d_lossy();
    test_error_reporting();
    std::printf("cuda available (via C ABI): %s\n", csa_cuda_available() ? "yes" : "no");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
