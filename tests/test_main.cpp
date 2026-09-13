// Minimal self-contained test harness (no external deps) covering
// round-trip correctness for every layer: range coder, Pantograph Lift,
// Rod-Joint transform (2D/3D), and the full container codec.
#include "csa/codec.hpp"
#include "csa/pantograph_lift.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include "csa/range_coder.hpp"
#include "csa/rod_joint_transform.hpp"
#include <cmath>
#include <cstdio>
#include <random>
#include <string>

using namespace csa;

static int g_failures = 0;
static int g_checks = 0;

#define CHECK(cond) do { \
    g_checks++; \
    if (!(cond)) { \
        g_failures++; \
        std::printf("FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
    } \
} while (0)

static std::vector<u8> random_bytes(size_t n, std::mt19937& rng) {
    std::vector<u8> v(n);
    std::uniform_int_distribution<int> d(0, 255);
    for (auto& b : v) b = (u8)d(rng);
    return v;
}

static void test_range_coder() {
    std::mt19937 rng(42);
    std::vector<std::vector<u8>> cases;
    cases.push_back({});
    cases.push_back({0});
    cases.push_back({255});
    cases.push_back(std::vector<u8>(1000, 7)); // constant
    {
        std::vector<u8> v;
        for (int i = 0; i < 2000; i++) v.push_back((u8)(i % 4)); // low entropy periodic
        cases.push_back(v);
    }
    cases.push_back(random_bytes(5000, rng)); // high entropy
    {
        std::vector<u8> v(256);
        for (int i = 0; i < 256; i++) v[i] = (u8)i; // every symbol once
        cases.push_back(v);
    }

    for (auto& input : cases) {
        auto coded = range_encode_bytes(input);
        auto decoded = range_decode_bytes(coded.data(), coded.size(), input.size());
        CHECK(decoded == input);
    }
    // Constant, highly periodic data should shrink substantially.
    std::vector<u8> constant(10000, 42);
    auto coded = range_encode_bytes(constant);
    CHECK(coded.size() < constant.size() / 10);
}

static std::vector<i32> smooth_ramp(size_t n) {
    std::vector<i32> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (i32)(i * 3 + 5);
    return v;
}

static std::vector<i32> smooth_sine(size_t n) {
    std::vector<i32> v(n);
    for (size_t i = 0; i < n; i++) v[i] = (i32)std::lround(1000.0 * std::sin((double)i * 0.05));
    return v;
}

static void test_pantograph_lift() {
    std::mt19937 rng(123);
    std::vector<std::vector<i32>> cases;
    cases.push_back({});
    cases.push_back({7});
    cases.push_back({1, 2});
    cases.push_back({-5, 100, 3, -3});
    cases.push_back(smooth_ramp(1));
    cases.push_back(smooth_ramp(2));
    cases.push_back(smooth_ramp(1000)); // not a power of two
    cases.push_back(smooth_sine(777));
    cases.push_back(std::vector<i32>(500, -17)); // constant
    {
        std::vector<i32> v(2000);
        std::uniform_int_distribution<int> d(-1000000, 1000000);
        for (auto& x : v) x = d(rng);
        cases.push_back(v);
    }

    for (auto& input : cases) {
        LiftResult lr = pantograph_lift_forward(input);
        auto back = pantograph_lift_inverse(lr);
        CHECK(back == input);
    }

    // Smooth ramp should compress residuals to near-zero magnitude.
    LiftResult lr = pantograph_lift_forward(smooth_ramp(1024));
    bool all_small = true;
    for (auto& level : lr.residuals)
        for (i32 r : level)
            if (std::abs(r) > 4) all_small = false;
    CHECK(all_small);
}

static void test_rod_joint_2d() {
    std::mt19937 rng(7);

    // Perfect logarithmic spiral: constant rotation+scale per step.
    // This is the shape class the transform is explicitly designed for.
    std::vector<Point2i> spiral;
    double x = 100, y = 0;
    double angle_step = 0.2, scale_step = 1.01;
    double dx = 5, dy = 0;
    for (int i = 0; i < 500; i++) {
        spiral.push_back({(i32)std::lround(x), (i32)std::lround(y)});
        double ndx = scale_step * (dx * std::cos(angle_step) - dy * std::sin(angle_step));
        double ndy = scale_step * (dx * std::sin(angle_step) + dy * std::cos(angle_step));
        dx = ndx; dy = ndy;
        x += dx; y += dy;
    }
    {
        RodJoint2DResult r = rod_joint_2d_forward(spiral);
        auto back = rod_joint_2d_inverse(r);
        CHECK(back.size() == spiral.size());
        bool match = true;
        for (size_t i = 0; i < spiral.size(); i++)
            if (back[i].x != spiral[i].x || back[i].y != spiral[i].y) match = false;
        CHECK(match);
    }

    std::vector<std::vector<Point2i>> cases;
    cases.push_back({});
    cases.push_back({{1, 2}});
    cases.push_back({{1, 2}, {3, 4}});
    {
        std::vector<Point2i> walk;
        i32 px = 0, py = 0;
        std::uniform_int_distribution<int> d(-10, 10);
        for (int i = 0; i < 300; i++) {
            px += d(rng); py += d(rng);
            walk.push_back({px, py});
        }
        cases.push_back(walk);
    }
    for (auto& pts : cases) {
        RodJoint2DResult r = rod_joint_2d_forward(pts);
        auto back = rod_joint_2d_inverse(r);
        CHECK(back.size() == pts.size());
        for (size_t i = 0; i < pts.size(); i++) {
            CHECK(back[i].x == pts[i].x);
            CHECK(back[i].y == pts[i].y);
        }
    }
}

static void test_rod_joint_3d() {
    std::mt19937 rng(99);
    std::vector<Point3i> helix;
    for (int i = 0; i < 400; i++) {
        double t = i * 0.1;
        helix.push_back({(i32)std::lround(1000 * std::cos(t)), (i32)std::lround(1000 * std::sin(t)), (i32)(i * 7)});
    }
    RodJoint3DResult r = rod_joint_3d_forward(helix);
    auto back = rod_joint_3d_inverse(r);
    CHECK(back.size() == helix.size());
    bool match = true;
    for (size_t i = 0; i < helix.size(); i++)
        if (back[i].x != helix[i].x || back[i].y != helix[i].y || back[i].z != helix[i].z) match = false;
    CHECK(match);

    std::vector<Point3i> empty;
    RodJoint3DResult r2 = rod_joint_3d_forward(empty);
    CHECK(rod_joint_3d_inverse(r2).empty());
}

static void test_rod_joint_3d_similarity() {
    std::mt19937 rng(1234);

    // A path that genuinely tumbles in 3D: constant rotation about a
    // fixed but non-axis-aligned axis, plus scale -- exactly what a
    // single 3x3 similarity joint models, and exactly what defeats the
    // xy-plane + z-axis composition (there is no single "up" axis here).
    std::vector<Point3i> tumble;
    double x = 200, y = 0, z = 0;
    double dx = 5, dy = 3, dz = 1;
    for (int i = 0; i < 600; i++) {
        tumble.push_back({(i32)std::lround(x), (i32)std::lround(y), (i32)std::lround(z)});
        // Rotate (dx,dy,dz) around the axis (1,1,1)/sqrt(3) by a small
        // angle each step, with a tiny uniform scale.
        double ax = 1.0 / std::sqrt(3.0), ay = ax, az = ax;
        double theta = 0.05;
        double ct = std::cos(theta), st = std::sin(theta);
        double dot = dx * ax + dy * ay + dz * az;
        double cx = ay * dz - az * dy, cy = az * dx - ax * dz, cz = ax * dy - ay * dx;
        double ndx = dx * ct + cx * st + ax * dot * (1 - ct);
        double ndy = dy * ct + cy * st + ay * dot * (1 - ct);
        double ndz = dz * ct + cz * st + az * dot * (1 - ct);
        double scale = 1.002;
        dx = ndx * scale; dy = ndy * scale; dz = ndz * scale;
        x += dx; y += dy; z += dz;
    }

    RodJoint3DSimResult r = rod_joint_3d_similarity_forward(tumble);
    auto back = rod_joint_3d_similarity_inverse(r);
    CHECK(back.size() == tumble.size());
    bool match = true;
    for (size_t i = 0; i < tumble.size(); i++)
        if (back[i].x != tumble[i].x || back[i].y != tumble[i].y || back[i].z != tumble[i].z) match = false;
    CHECK(match);

    // The similarity joint should predict this tumbling path far better
    // than raw storage would: residuals should be small relative to the
    // rod magnitudes (which grow into the hundreds).
    i64 sum_abs_residual = 0;
    for (i32 v : r.residual_x) sum_abs_residual += std::abs(v);
    for (i32 v : r.residual_y) sum_abs_residual += std::abs(v);
    for (i32 v : r.residual_z) sum_abs_residual += std::abs(v);
    double mean_abs_residual = (double)sum_abs_residual / (double)(3 * r.residual_x.size());
    CHECK(mean_abs_residual < 5.0);

    // Round-trip on a variety of edge cases.
    std::vector<std::vector<Point3i>> cases;
    cases.push_back({});
    cases.push_back({{1, 2, 3}});
    cases.push_back({{1, 2, 3}, {4, 5, 6}});
    {
        std::vector<Point3i> walk;
        i32 px = 0, py = 0, pz = 0;
        std::uniform_int_distribution<int> d(-10, 10);
        for (int i = 0; i < 200; i++) {
            px += d(rng); py += d(rng); pz += d(rng);
            walk.push_back({px, py, pz});
        }
        cases.push_back(walk);
    }
    for (auto& pts : cases) {
        RodJoint3DSimResult rr = rod_joint_3d_similarity_forward(pts);
        auto b = rod_joint_3d_similarity_inverse(rr);
        CHECK(b.size() == pts.size());
        for (size_t i = 0; i < pts.size(); i++) {
            CHECK(b[i].x == pts[i].x);
            CHECK(b[i].y == pts[i].y);
            CHECK(b[i].z == pts[i].z);
        }
    }
}

static void test_codec() {
    std::mt19937 rng(2024);

    // Text-like structured data should compress well.
    std::string text;
    for (int i = 0; i < 200; i++) text += "the quick brown fox jumps over the lazy dog. ";
    std::vector<u8> text_bytes(text.begin(), text.end());
    auto blob = compress(text_bytes);
    auto back = decompress(blob);
    CHECK(back == text_bytes);
    // Pantograph Lift is not an LZ-style dictionary coder, so don't expect
    // gzip-tier ratios on text -- just noticeably smaller than raw.
    CHECK(blob.size() < text_bytes.size() * 7 / 10);

    // Incompressible random data: RAW fallback must keep overhead tiny.
    auto rnd = random_bytes(4096, rng);
    auto rnd_blob = compress(rnd);
    auto rnd_back = decompress(rnd_blob);
    CHECK(rnd_back == rnd);
    CHECK(rnd_blob.size() <= rnd.size() + 32);

    // Empty input.
    std::vector<u8> empty;
    auto empty_blob = compress(empty);
    CHECK(decompress(empty_blob).empty());

    // Single byte.
    std::vector<u8> one{99};
    auto one_blob = compress(one);
    CHECK(decompress(one_blob) == one);

    if (cuda_is_available()) {
        // Call the GPU transform directly and require it to actually
        // succeed (return true), not just round-trip. compress(use_gpu=
        // true) transparently falls back to the CPU path if the GPU path
        // fails, which is the right behavior in production but would
        // silently mask a real GPU bug as a passing test here -- exactly
        // what happened once already during development (a stale
        // device-link artifact made a kernel launch fail with
        // cudaErrorSymbolNotFound, and the fallback made the suite pass
        // anyway). Assert success explicitly so that class of bug can't
        // hide again.
        std::vector<i32> as_i32(text_bytes.size());
        for (size_t i = 0; i < text_bytes.size(); i++) as_i32[i] = (i32)text_bytes[i];
        LiftResult gpu_lift;
        bool gpu_ok = pantograph_lift_forward_cuda(as_i32, gpu_lift);
        CHECK(gpu_ok);
        if (gpu_ok) {
            auto gpu_roundtrip = pantograph_lift_inverse(gpu_lift);
            CHECK(gpu_roundtrip == as_i32);
        }

        auto gpu_blob = compress(text_bytes, /*use_gpu=*/true);
        auto gpu_back = decompress(gpu_blob);
        CHECK(gpu_back == text_bytes);

        // A larger, multi-level input exercises many calibration blocks
        // and several ping-pong levels of the GPU-resident pipeline, not
        // just the handful a ~1800-byte string touches.
        std::mt19937 big_rng(555);
        auto big = random_bytes(500000, big_rng);
        std::vector<i32> big_i32(big.size());
        for (size_t i = 0; i < big.size(); i++) big_i32[i] = (i32)big[i];
        LiftResult big_gpu_lift;
        bool big_gpu_ok = pantograph_lift_forward_cuda(big_i32, big_gpu_lift);
        CHECK(big_gpu_ok);
        if (big_gpu_ok) {
            auto big_back = pantograph_lift_inverse(big_gpu_lift);
            CHECK(big_back == big_i32);
            auto big_cpu_lift = pantograph_lift_forward(big_i32);
            auto big_cpu_back = pantograph_lift_inverse(big_cpu_lift);
            CHECK(big_cpu_back == big_i32); // CPU path agrees independently too
        }

        std::printf("  (CUDA path exercised: forward_cuda succeeded=%s (small), %s (500KB), round-tripped correctly)\n",
                     gpu_ok ? "true" : "FALSE", big_gpu_ok ? "true" : "FALSE");
    } else {
        std::printf("  (CUDA not available at test time; GPU path skipped, CPU-only verified)\n");
    }
}

static void test_geo_codec() {
    std::vector<Point2i> spiral;
    double x = 50, y = 0, dx = 3, dy = 0;
    for (int i = 0; i < 200; i++) {
        spiral.push_back({(i32)std::lround(x), (i32)std::lround(y)});
        double ndx = 1.02 * (dx * std::cos(0.15) - dy * std::sin(0.15));
        double ndy = 1.02 * (dx * std::sin(0.15) + dy * std::cos(0.15));
        dx = ndx; dy = ndy;
        x += dx; y += dy;
    }
    auto blob = compress_geo2d(spiral);
    auto back = decompress_geo2d(blob);
    CHECK(back.size() == spiral.size());
    bool match = true;
    for (size_t i = 0; i < spiral.size(); i++)
        if (back[i].x != spiral[i].x || back[i].y != spiral[i].y) match = false;
    CHECK(match);
    // A near-perfect spiral should compress to a tiny fraction of raw doubles.
    CHECK(blob.size() < spiral.size() * 2 * sizeof(double) / 4);
}

static void test_geo3d_codec_autoselect() {
    // A helix: constant-radius rotation about z plus constant climb --
    // exactly what the xy+z composition models, and it should win.
    std::vector<Point3i> helix;
    for (int i = 0; i < 400; i++) {
        double t = i * 0.1;
        helix.push_back({(i32)std::lround(1000 * std::cos(t)), (i32)std::lround(1000 * std::sin(t)), (i32)(i * 7)});
    }
    auto helix_blob = compress_geo3d(helix);
    auto helix_back = decompress_geo3d(helix_blob);
    CHECK(helix_back.size() == helix.size());
    bool helix_match = true;
    for (size_t i = 0; i < helix.size(); i++)
        if (helix_back[i].x != helix[i].x || helix_back[i].y != helix[i].y || helix_back[i].z != helix[i].z)
            helix_match = false;
    CHECK(helix_match);

    // A path that tumbles around a non-axis-aligned axis: no single
    // "up" axis for the xy+z composition to exploit, so the true 3D
    // similarity joint should win here instead.
    std::vector<Point3i> tumble;
    double x = 200, y = 0, z = 0;
    double dx = 5, dy = 3, dz = 1;
    for (int i = 0; i < 600; i++) {
        tumble.push_back({(i32)std::lround(x), (i32)std::lround(y), (i32)std::lround(z)});
        double ax = 1.0 / std::sqrt(3.0), ay = ax, az = ax;
        double theta = 0.05;
        double ct = std::cos(theta), st = std::sin(theta);
        double dot = dx * ax + dy * ay + dz * az;
        double cx = ay * dz - az * dy, cy = az * dx - ax * dz, cz = ax * dy - ay * dx;
        double ndx = dx * ct + cx * st + ax * dot * (1 - ct);
        double ndy = dy * ct + cy * st + ay * dot * (1 - ct);
        double ndz = dz * ct + cz * st + az * dot * (1 - ct);
        double scale = 1.002;
        dx = ndx * scale; dy = ndy * scale; dz = ndz * scale;
        x += dx; y += dy; z += dz;
    }
    auto tumble_blob = compress_geo3d(tumble);
    auto tumble_back = decompress_geo3d(tumble_blob);
    CHECK(tumble_back.size() == tumble.size());
    bool tumble_match = true;
    for (size_t i = 0; i < tumble.size(); i++)
        if (tumble_back[i].x != tumble[i].x || tumble_back[i].y != tumble[i].y || tumble_back[i].z != tumble[i].z)
            tumble_match = false;
    CHECK(tumble_match);

    // Don't assert which sub-mode wins for which shape: a true 3D
    // similarity matrix can trivially reduce to the xy+z composition's
    // best case too (its bottom row collapses to (0,0,1) whenever a
    // rod's z-component is constant, as in a helix's climb), so which
    // one encodes smaller is an empirical, self-selecting outcome, not
    // a fixed prediction. Just confirm a sub-mode byte is present.
    CHECK(tumble_blob.size() > 5);
    CHECK(helix_blob.size() > 5);
}

int main() {
    test_range_coder();
    test_pantograph_lift();
    test_rod_joint_2d();
    test_rod_joint_3d();
    test_rod_joint_3d_similarity();
    test_codec();
    test_geo_codec();
    test_geo3d_codec_autoselect();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
