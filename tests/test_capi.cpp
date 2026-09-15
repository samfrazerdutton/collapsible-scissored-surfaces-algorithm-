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

static void test_geo3d_lossy() {
    std::vector<int32_t> xyz;
    double x = 0, y = 0, z = 0, heading = 0;
    unsigned int seed = 314159;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    for (int i = 0; i < 2000; i++) {
        heading += (next_rand() - 0.5) * 0.08;
        double speed = 8.0 + (next_rand() - 0.5);
        x += speed * std::cos(heading);
        y += speed * std::sin(heading);
        z += 3.0 + (next_rand() - 0.5) * 0.5;
        xyz.push_back((int32_t)(x + 0.5));
        xyz.push_back((int32_t)(y + 0.5));
        xyz.push_back((int32_t)(z + 0.5));
    }
    size_t count = xyz.size() / 3;

    csa_buffer lossless = csa_compress_geo3d(xyz.data(), count);
    csa_buffer lossy = csa_compress_geo3d_lossy(xyz.data(), count, /*quant_step=*/20, /*resync_interval=*/64);
    CHECK(lossy.size > 0);
    CHECK(lossy.size < lossless.size);

    size_t out_count = 0;
    csa_buffer decoded = csa_decompress_geo3d(lossy.data, lossy.size, &out_count);
    CHECK(out_count == count);
    if (decoded.data) {
        const int32_t* out_xyz = (const int32_t*)decoded.data;
        int32_t max_err = 0;
        for (size_t i = 0; i < xyz.size(); i++) {
            int32_t d = out_xyz[i] - xyz[i];
            if (d < 0) d = -d;
            if (d > max_err) max_err = d;
        }
        CHECK(max_err <= 20 * 64);
    }

    csa_free_buffer(lossless);
    csa_free_buffer(lossy);
    csa_free_buffer(decoded);
}

static void test_pose_lossy() {
    // A drone-circling-while-yawing 6-DOF pose stream: constant-radius xy
    // rotation + noisy steady climb (position) with orientation tracking
    // the heading + noise (quaternion), same shape class as csa_tests'
    // test_pose_codec.
    std::vector<int32_t> pose7;
    const double qscale = 1 << 20;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    unsigned int seed = 4242;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    double z = 0.0;
    for (int i = 0; i < 1500; i++) {
        double t = i * 0.05;
        pose7.push_back((int32_t)std::lround(2000 * std::cos(t)));
        pose7.push_back((int32_t)std::lround(2000 * std::sin(t)));
        z += 4.0 + (next_rand() - 0.5) * 0.5;
        pose7.push_back((int32_t)std::lround(z));
        pose7.push_back((int32_t)std::lround(qw * qscale));
        pose7.push_back((int32_t)std::lround(qx * qscale));
        pose7.push_back((int32_t)std::lround(qy * qscale));
        pose7.push_back((int32_t)std::lround(qz * qscale));

        double deg = 2.0 + (next_rand() - 0.5) * 0.3;
        double half = (deg * 3.14159265358979323846 / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw = qw * dqw - qx * 0 - qy * 0 - qz * dqz;
        double nx = qw * 0 + qx * dqw + qy * dqz - qz * 0;
        double ny = qw * 0 - qx * dqz + qy * dqw + qz * 0;
        double nz = qw * dqz + qx * 0 - qy * 0 + qz * dqw;
        qw = nw; qx = nx; qy = ny; qz = nz;
    }
    size_t count = pose7.size() / 7;

    csa_buffer lossless = csa_compress_pose(pose7.data(), count);
    CHECK(lossless.data != nullptr && lossless.size > 0);

    size_t out_count = 0;
    csa_buffer decoded = csa_decompress_pose(lossless.data, lossless.size, &out_count);
    CHECK(out_count == count);
    CHECK(decoded.data != nullptr &&
          std::memcmp(decoded.data, pose7.data(), pose7.size() * sizeof(int32_t)) == 0);

    csa_buffer lossy = csa_compress_pose_lossy(pose7.data(), count, /*pos_quant=*/8, /*pos_resync=*/64,
                                                /*quat_quant=*/32, /*quat_resync=*/32);
    CHECK(lossy.size > 0 && lossy.size < lossless.size);

    size_t lossy_out_count = 0;
    csa_buffer lossy_decoded = csa_decompress_pose(lossy.data, lossy.size, &lossy_out_count);
    CHECK(lossy_out_count == count);
    if (lossy_decoded.data) {
        const int32_t* out = (const int32_t*)lossy_decoded.data;
        int32_t max_err = 0;
        for (size_t i = 0; i < pose7.size(); i++) {
            int32_t d = out[i] - pose7[i];
            if (d < 0) d = -d;
            if (d > max_err) max_err = d;
        }
        CHECK(max_err <= 64 * 64); // generous bound, same spirit as the geo3d lossy check
    }

    csa_free_buffer(lossless);
    csa_free_buffer(decoded);
    csa_free_buffer(lossy);
    csa_free_buffer(lossy_decoded);
}

static void test_rans() {
    std::vector<unsigned char> input;
    for (int i = 0; i < 20000; i++) input.push_back((unsigned char)((i * 37) % 7)); // skewed, low-entropy

    csa_buffer coded = csa_rans_encode(input.data(), input.size(), /*num_lanes=*/4, /*scale_bits=*/14);
    CHECK(coded.data != nullptr && coded.size > 0 && coded.size < input.size() / 2);

    csa_buffer decoded = csa_rans_decode(coded.data, coded.size);
    CHECK(decoded.size == input.size());
    CHECK(decoded.data != nullptr && std::memcmp(decoded.data, input.data(), input.size()) == 0);

    csa_free_buffer(coded);
    csa_free_buffer(decoded);

    // Empty input round-trips too: encode still emits its small fixed
    // header (so `coded` is non-empty), but decoding it back yields an
    // empty result.
    csa_buffer empty_coded = csa_rans_encode(nullptr, 0, 4, 14);
    CHECK(empty_coded.data != nullptr && empty_coded.size > 0);
    csa_buffer empty_decoded = csa_rans_decode(empty_coded.data, empty_coded.size);
    CHECK(empty_decoded.data == nullptr && empty_decoded.size == 0);
    csa_free_buffer(empty_coded);
    csa_free_buffer(empty_decoded);
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
    test_geo3d_lossy();
    test_pose_lossy();
    test_rans();
    test_error_reporting();
    std::printf("cuda available (via C ABI): %s\n", csa_cuda_available() ? "yes" : "no");
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
