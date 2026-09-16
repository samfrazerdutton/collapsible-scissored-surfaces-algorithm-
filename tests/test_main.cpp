// Minimal self-contained test harness (no external deps) covering
// round-trip correctness for every layer: range coder, Pantograph Lift,
// Rod-Joint transform (2D/3D), and the full container codec.
#include "csa/bwt_codec.hpp"
#include "csa/bwt_transform.hpp"
#include "csa/codec.hpp"
#include "csa/lz_codec.hpp"
#include "csa/lz_matcher.hpp"
#include "csa/pantograph_lift.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include "csa/pose_stream.hpp"
#include "csa/quaternion_calibration_cuda.hpp"
#include "csa/quaternion_joint.hpp"
#include "csa/range_coder.hpp"
#include "csa/rans_coder.hpp"
#include "csa/rod_joint_transform.hpp"
#include "csa/simd.hpp"
#include "csa/simd_cuda.hpp"
#include "csa/thread_pool.hpp"
#include "csa/work_stealing_pool.hpp"
#include <algorithm>
#include <chrono>
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

static void test_lz_matcher() {
    std::mt19937 rng(77);

    std::vector<std::vector<u8>> cases;
    cases.push_back({});
    cases.push_back({1});
    cases.push_back({1, 2, 3});
    cases.push_back(std::vector<u8>(500, 7)); // maximally repetitive
    cases.push_back(random_bytes(2000, rng)); // no exploitable structure
    {
        // Overlapping-match case: distance < length forces a byte-by-byte
        // overlap copy on reconstruction, not a plain memcpy.
        std::vector<u8> v;
        for (int i = 0; i < 300; i++) v.push_back((u8)('a' + (i % 3)));
        cases.push_back(v);
    }
    {
        std::string s;
        for (int i = 0; i < 300; i++) s += "the quick brown fox jumps over the lazy dog. ";
        cases.push_back(std::vector<u8>(s.begin(), s.end()));
    }

    for (auto& input : cases) {
        auto tokens = lz_parse(input);
        auto back = lz_tokens_reconstruct(tokens);
        CHECK(back == input);
    }

    // The maximally repetitive case should collapse to very few tokens
    // (one literal to seed the dictionary, then one giant match), proving
    // the matcher isn't just emitting literals.
    auto rep_tokens = lz_parse(std::vector<u8>(500, 7));
    CHECK(rep_tokens.size() < 5);
}

static void test_lz_codec() {
    std::mt19937 rng(88);

    std::vector<std::vector<u8>> cases;
    cases.push_back({});
    cases.push_back({42});
    cases.push_back(std::vector<u8>(1000, 9));
    cases.push_back(random_bytes(3000, rng));

    std::string text;
    for (int i = 0; i < 500; i++) text += "the quick brown fox jumps over the lazy dog. ";
    cases.push_back(std::vector<u8>(text.begin(), text.end()));

    for (auto& input : cases) {
        auto blob = lz_encode(input);
        size_t pos = 0;
        auto back = lz_decode(blob.data(), blob.size(), pos);
        CHECK(back == input);
        CHECK(pos == blob.size()); // decode must consume exactly its own blob
    }

    // Highly repetitive text should compress to a tiny fraction of its
    // size -- this is the entire point of adding a dictionary matcher.
    auto text_bytes = std::vector<u8>(text.begin(), text.end());
    auto text_blob = lz_encode(text_bytes);
    CHECK(text_blob.size() < text_bytes.size() / 20);
}

static void test_bwt_transform() {
    // The classic textbook example: BWT("banana") is well known, and
    // -- more importantly for this implementation -- hand-verifiable:
    // the sentinel-based construction here was derived and checked by
    // hand against this exact case before being trusted with anything
    // larger. This test is that derivation, permanently encoded.
    {
        std::vector<u8> banana = {'b', 'a', 'n', 'a', 'n', 'a'};
        BwtBlockResult r = bwt_encode_block(banana);
        CHECK(r.symbols.size() == banana.size() + 1);
        auto back = bwt_decode_block(r.symbols);
        CHECK(back == banana);
    }

    std::mt19937 rng(2025);
    std::vector<std::vector<u8>> cases;
    cases.push_back({1}); // single byte
    cases.push_back(std::vector<u8>(500, 7)); // maximally repetitive: stresses the
                                               // sentinel's job of keeping every
                                               // suffix distinct despite total periodicity
    cases.push_back(random_bytes(5000, rng)); // no exploitable structure
    {
        std::vector<u8> all_bytes(256);
        for (int i = 0; i < 256; i++) all_bytes[i] = (u8)i;
        cases.push_back(all_bytes); // every possible byte value, once each
    }
    {
        std::string s;
        for (int i = 0; i < 300; i++) s += "the quick brown fox jumps over the lazy dog. ";
        cases.push_back(std::vector<u8>(s.begin(), s.end()));
    }

    for (auto& block : cases) {
        BwtBlockResult r = bwt_encode_block(block);
        CHECK(r.symbols.size() == block.size() + 1);
        // The sentinel (0) must appear in the output exactly once.
        size_t zero_count = 0;
        for (u16 s : r.symbols) if (s == 0) zero_count++;
        CHECK(zero_count == 1);
        auto back = bwt_decode_block(r.symbols);
        CHECK(back == block);
    }
}

// SA-IS (linear-time suffix array construction) is intricate enough that
// it's trusted through exhaustive cross-validation against the already-
// verified O(n log^2 n) prefix-doubling implementation, not just a
// handful of hand-checked examples -- deliberately including tiny
// alphabets (which force deep recursion, the part of SA-IS most likely
// to hide a subtle bug) and maximally periodic inputs.
static void test_bwt_sais_matches_reference() {
    // The hand-derived "banana" suffix array from this session's BWT
    // design work, cross-checked against both implementations: symbolic
    // values b=2, a=1, n=3, with the usual unique-smallest sentinel (0)
    // appended. Verified by hand: SA = [6, 5, 3, 1, 0, 4, 2].
    {
        std::vector<int> t = {2, 1, 3, 1, 3, 1, 0};
        std::vector<int> expected = {6, 5, 3, 1, 0, 4, 2};
        CHECK(bwt_debug_build_suffix_array_reference(t) == expected);
        CHECK(bwt_debug_build_suffix_array_sais(t, 4) == expected);
    }

    std::mt19937 rng(31415);
    auto run_case = [&](const std::vector<int>& t, int alphabet_size) {
        auto ref = bwt_debug_build_suffix_array_reference(t);
        auto sais = bwt_debug_build_suffix_array_sais(t, alphabet_size);
        CHECK(ref == sais);
    };

    // Tiny alphabets over a spread of random lengths: guarantees lots of
    // repeated LMS substrings, forcing the recursive reduction path
    // (not just the "all names already distinct, no recursion" path).
    for (int alphabet_size : {2, 3, 4, 8, 257}) {
        for (int trial = 0; trial < 30; trial++) {
            std::uniform_int_distribution<int> len_dist(1, 400);
            int n = len_dist(rng);
            std::vector<int> t(n);
            std::uniform_int_distribution<int> sym_dist(1, alphabet_size - 1);
            for (int i = 0; i < n - 1; i++) t[i] = sym_dist(rng);
            t[n - 1] = 0;
            run_case(t, alphabet_size);
        }
    }

    // Larger, realistic-shaped inputs matching bwt_encode_block's actual
    // convention exactly (real byte values 1..256, sentinel 0).
    for (int trial = 0; trial < 10; trial++) {
        std::uniform_int_distribution<int> len_dist(500, 4000);
        int n = len_dist(rng);
        std::vector<int> t(n);
        std::uniform_int_distribution<int> sym_dist(1, 256);
        for (int i = 0; i < n - 1; i++) t[i] = sym_dist(rng);
        t[n - 1] = 0;
        run_case(t, 257);
    }

    // Maximal recursion stress: a periodic pattern and a fully constant
    // block (the classic BWT edge case), both over the real alphabet.
    {
        std::vector<int> t(2000);
        for (int i = 0; i < 1999; i++) t[i] = 1 + (i % 3);
        t[1999] = 0;
        run_case(t, 257);
    }
    {
        std::vector<int> t(1500, 5);
        t[1499] = 0;
        run_case(t, 257);
    }

    // Degenerate small sizes.
    run_case({0}, 257);
    run_case({5, 0}, 257);
}

static void test_bwt_codec() {
    std::mt19937 rng(2026);

    std::vector<std::vector<u8>> cases;
    cases.push_back({});
    cases.push_back({42});
    cases.push_back(std::vector<u8>(1000, 9));
    cases.push_back(random_bytes(3000, rng));

    std::string text;
    for (int i = 0; i < 500; i++) text += "the quick brown fox jumps over the lazy dog. ";
    cases.push_back(std::vector<u8>(text.begin(), text.end()));

    // Spans multiple BWT blocks (kBwtDefaultBlockSize is 256KB): exercises
    // the block-chunking logic itself, not just a single block's transform.
    cases.push_back(random_bytes(kBwtDefaultBlockSize * 2 + 12345, rng));

    // Forces the RUNA/RUNB zero-run decoder's trickiest case: a run of
    // MTF rank-0 that extends all the way to the exact end of a block,
    // with no real (non-run) symbol following it in that block at all --
    // the decoder has to recognize "the pending run alone already
    // reaches this block's known length" without trying to read another
    // entropy symbol first (which would otherwise consume the start of
    // the next block's stream). A full random block followed by a
    // second block that's entirely one repeated byte guarantees the
    // second block's MTF output ends in a single giant zero-run.
    {
        auto boundary_case = random_bytes(kBwtDefaultBlockSize, rng);
        auto tail = std::vector<u8>(5000, (u8)'Z');
        boundary_case.insert(boundary_case.end(), tail.begin(), tail.end());
        cases.push_back(boundary_case);
    }

    for (auto& input : cases) {
        auto blob = bwt_encode(input);
        size_t pos = 0;
        auto back = bwt_decode(blob.data(), blob.size(), pos);
        CHECK(back == input);
        CHECK(pos == blob.size());
    }

    // Highly repetitive text should compress to a tiny fraction of its
    // size -- BWT+MTF on a single repeated sentence collapses almost
    // entirely to rank-0 after the first occurrence.
    auto text_bytes = std::vector<u8>(text.begin(), text.end());
    auto text_blob = bwt_encode(text_bytes);
    CHECK(text_blob.size() < text_bytes.size() / 20);
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

// Lossy Pantograph Lift: quant_step <= 1 must be exactly lossless (the
// same closed-loop-quantization invariant every other lossy mode in this
// codebase provides), and quant_step > 1 must still round-trip to a
// *bounded* error while compressing measurably smaller. Unlike Rod-
// Joint's lossy modes (a flat sequence of rods, where error accumulates
// as a simple random walk with a clean per-rod bound), error here can
// compound across this transform's O(log n) cascaded levels -- a
// quantized residual at one level perturbs the "update" value that
// becomes part of the *next* level's own input, which then gets
// quantized again. There's no clean closed-form bound to assert a priori
// the way rod_joint_2d_error_bound() provides one, so this test measures
// the actual observed error on representative inputs and prints it,
// rather than asserting an unverified formula.
static void test_pantograph_lift_lossy() {
    // quant_step <= 1 is exactly lossless, on every existing case.
    for (auto& input : {smooth_ramp(1000), smooth_sine(777), std::vector<i32>(500, -17)}) {
        LiftResult lr0 = pantograph_lift_forward(input, 0); // 0 normalizes to 1
        LiftResult lr1 = pantograph_lift_forward(input, 1);
        CHECK(pantograph_lift_inverse(lr0) == input);
        CHECK(pantograph_lift_inverse(lr1) == input);
    }

    // A smooth-but-noisy signal (like a real sensor reading), the
    // realistic target for this mode -- not a perfectly smooth ramp,
    // which would trivially compress either way.
    std::vector<i32> signal(4096);
    unsigned int seed = 2027;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    double v = 0.0;
    for (size_t i = 0; i < signal.size(); i++) {
        v += 3.0 + (next_rand() - 0.5) * 2.0; // steady drift with noise, like the z-axis of a climbing path
        signal[i] = (i32)std::lround(v);
    }

    u32 quant_step = 16;
    LiftResult lossless = pantograph_lift_forward(signal, 1);
    LiftResult lossy = pantograph_lift_forward(signal, quant_step);
    auto back = pantograph_lift_inverse(lossy);
    CHECK(back.size() == signal.size());

    i32 max_err = 0;
    for (size_t i = 0; i < signal.size(); i++) max_err = std::max(max_err, std::abs(signal[i] - back[i]));

    // A generous, empirically-motivated bound: quant_step/2 per level
    // compounding across at most log2(n) levels, doubled again as
    // headroom rather than a claimed tight derivation.
    int levels = 0;
    for (size_t n = signal.size(); n > 1; n /= 2) levels++;
    i32 generous_bound = (i32)quant_step * (levels + 1);
    CHECK(max_err <= generous_bound);

    std::printf("  (Pantograph Lift lossy: quant_step=%u, max observed error=%d (bound %d))\n",
                 quant_step, max_err, generous_bound);

    // Real, end-to-end compression gain through the actual public API:
    // compress_geo3d_lossy now threads quant_step through the xy+z
    // composition model's z-axis Pantograph Lift too (previously it
    // could only go lossy via the true 3D similarity joint). A helix-
    // like path (constant-radius rotation + steady z climb) is exactly
    // the shape compress_geo3d's auto-select already favors the
    // composition model for (see test_geo3d_codec_autoselect), so this
    // is testing the composition model's new lossy z-axis specifically,
    // not just whichever model happens to win.
    std::vector<Point3i> helix;
    unsigned int hseed = 2028;
    auto next_rand2 = [&]() { hseed = hseed * 1664525u + 1013904223u; return (double)(hseed >> 8) / (double)(1u << 24); };
    double hz = 0.0;
    for (int i = 0; i < 2000; i++) {
        double t = i * 0.1;
        hz += 3.0 + (next_rand2() - 0.5) * 0.5;
        helix.push_back({(i32)std::lround(1000 * std::cos(t)), (i32)std::lround(1000 * std::sin(t)), (i32)std::lround(hz)});
    }

    auto helix_lossless = compress_geo3d(helix);
    auto helix_lossy = compress_geo3d_lossy(helix, quant_step, 64);
    CHECK(helix_lossy.size() < helix_lossless.size());

    auto helix_back = decompress_geo3d(helix_lossy);
    CHECK(helix_back.size() == helix.size());
    i32 helix_max_err = 0;
    for (size_t i = 0; i < helix.size(); i++)
        helix_max_err = std::max({helix_max_err, std::abs(helix[i].x - helix_back[i].x),
                                   std::abs(helix[i].y - helix_back[i].y), std::abs(helix[i].z - helix_back[i].z)});
    CHECK(helix_max_err > 0); // genuinely lossy, not accidentally exact

    auto helix_exact_blob = compress_geo3d_lossy(helix, 1, 0);
    auto helix_exact_back = decompress_geo3d(helix_exact_blob);
    bool helix_exact_match = helix_exact_back.size() == helix.size();
    for (size_t i = 0; helix_exact_match && i < helix.size(); i++)
        if (helix_exact_back[i].x != helix[i].x || helix_exact_back[i].y != helix[i].y || helix_exact_back[i].z != helix[i].z)
            helix_exact_match = false;
    CHECK(helix_exact_match);

    std::printf("  (Geo3D composition lossy: %zu -> %zu bytes, max coordinate error = %d)\n",
                 helix_lossless.size(), helix_lossy.size(), helix_max_err);
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
    // compress() now also tries the LZ dictionary-matcher candidate, which
    // should win handily on this maximally repetitive text and auto-select
    // itself -- a much stronger bar than the old Pantograph-Lift-only
    // "noticeably smaller than raw" expectation.
    CHECK(blob.size() < text_bytes.size() / 10);

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

        // CudaLiftSession must be a drop-in, bit-for-bit equivalent to the
        // one-shot pantograph_lift_forward_cuda() -- reusing device
        // buffers must never change the numeric result. Drive it through
        // small -> large -> small again, specifically to exercise both
        // directions of capacity change: growing buffers (small -> large)
        // and reusing an already-larger buffer for a smaller call
        // (large -> small) without the unused tail of the buffer leaking
        // into the result.
        {
            CudaLiftSession session;
            LiftResult s_small1;
            bool ok1 = session.forward(as_i32, s_small1);
            CHECK(ok1);
            if (ok1) {
                CHECK(s_small1.residuals.size() == gpu_lift.residuals.size());
                bool same = true;
                for (size_t lvl = 0; lvl < s_small1.residuals.size() && same; lvl++)
                    if (s_small1.residuals[lvl] != gpu_lift.residuals[lvl]) same = false;
                CHECK(same);
                CHECK(s_small1.base == gpu_lift.base);
                auto back1 = pantograph_lift_inverse(s_small1);
                CHECK(back1 == as_i32);
            }

            LiftResult s_big;
            bool ok2 = session.forward(big_i32, s_big);
            CHECK(ok2);
            if (ok2) {
                auto back_big = pantograph_lift_inverse(s_big);
                CHECK(back_big == big_i32);
            }

            // Same session, back to the small input -- reuses the larger
            // buffers allocated for big_i32.
            LiftResult s_small2;
            bool ok3 = session.forward(as_i32, s_small2);
            CHECK(ok3);
            if (ok3) {
                CHECK(s_small2.residuals.size() == gpu_lift.residuals.size());
                bool same = true;
                for (size_t lvl = 0; lvl < s_small2.residuals.size() && same; lvl++)
                    if (s_small2.residuals[lvl] != gpu_lift.residuals[lvl]) same = false;
                CHECK(same);
                CHECK(s_small2.base == gpu_lift.base);
                auto back2 = pantograph_lift_inverse(s_small2);
                CHECK(back2 == as_i32);
            }

            std::printf("  (CudaLiftSession exercised: small->big->small all succeeded=%s, bit-identical to one-shot forward_cuda)\n",
                         (ok1 && ok2 && ok3) ? "true" : "FALSE");
        }
    } else {
        std::printf("  (CUDA not available at test time; GPU path skipped, CPU-only verified)\n");
    }
}

// Regression guard for the adaptive Pantograph-Lift skip: a large,
// highly repetitive input should still round-trip correctly and land on
// the LZ candidate (mode byte 4) even when Pantograph Lift was skipped.
static void test_codec_adaptive_skip() {
    std::string unit = "the quick brown fox jumps over the lazy dog. ";
    std::string text;
    while (text.size() < 200000) text += unit; // well past kAdaptiveSizeThreshold
    std::vector<u8> input(text.begin(), text.end());

    auto blob = compress(input);
    CHECK(blob.size() > 5);
    CHECK(blob[4] == 4); // Mode::GeneralLZ
    auto back = decompress(blob);
    CHECK(back == input);
    CHECK(blob.size() < input.size() / 20); // still compresses excellently
}

// The lossy Rod-Joint path: quantized residuals with periodic exact
// resync to bound drift. Checks the three properties that actually
// matter -- per-coordinate error stays within the documented bound,
// resync genuinely limits how far absolute error can drift, and lossy
// mode actually compresses better than lossless (otherwise what's the
// point) -- not just "it doesn't crash".
static void test_rod_joint_2d_lossy() {
    std::mt19937 rng(4242);

    // A believable noisy path (not a perfect analytic curve): lossy mode
    // should help most on exactly this kind of real-world-ish data.
    std::vector<Point2i> path;
    double x = 0, y = 0, heading = 0;
    for (int i = 0; i < 4000; i++) {
        heading += std::uniform_real_distribution<double>(-0.05, 0.05)(rng);
        double speed = 8.0 + std::uniform_real_distribution<double>(-0.5, 0.5)(rng);
        x += speed * std::cos(heading);
        y += speed * std::sin(heading);
        path.push_back({(i32)std::lround(x), (i32)std::lround(y)});
    }

    u32 quant_step = 20;
    u32 resync_interval = 64;
    RodJoint2DResult r = rod_joint_2d_forward(path, /*lag=*/1, quant_step, resync_interval);
    auto back = rod_joint_2d_inverse(r);
    CHECK(back.size() == path.size());
    u32 bound = rod_joint_2d_error_bound(quant_step);

    // Per-rod reconstruction error bound applies to *non-resync* rods --
    // a resync rod deliberately absorbs whatever drift accumulated since
    // the last resync (that's the whole mechanism), so its own rod value
    // can differ a lot from the true rod; that's expected, not a bug.
    // Non-resync rods still individually track the true rod tightly.
    bool nonresync_rods_within_bound = true;
    for (size_t i = 1; i < path.size(); i++) {
        bool is_resync = (i % resync_interval) == 0;
        if (is_resync) continue;
        i32 true_rod_x = path[i].x - path[i - 1].x;
        i32 true_rod_y = path[i].y - path[i - 1].y;
        i32 rec_rod_x = back[i].x - back[i - 1].x;
        i32 rec_rod_y = back[i].y - back[i - 1].y;
        // Allow a small rounding slack (+1) since the bound is
        // approximate for odd quant_step, not a razor-exact guarantee.
        if ((u32)std::abs(true_rod_x - rec_rod_x) > bound + 1) nonresync_rods_within_bound = false;
        if ((u32)std::abs(true_rod_y - rec_rod_y) > bound + 1) nonresync_rods_within_bound = false;
    }
    CHECK(nonresync_rods_within_bound);

    // Resync actually resets drift: absolute position must be *exact*
    // right at each resync point (that's the entire point of the fix --
    // making the rod land on the true absolute point, not just on the
    // true rod added to an already-drifted position), and error between
    // resyncs must stay bounded rather than growing across the whole path.
    bool resync_points_exact = true;
    i32 max_abs_error = 0;
    for (size_t i = 0; i < path.size(); i++) {
        i32 ex = std::abs(path[i].x - back[i].x);
        i32 ey = std::abs(path[i].y - back[i].y);
        max_abs_error = std::max({max_abs_error, ex, ey});
        if (i > 0 && (i % resync_interval) == 0 && (ex != 0 || ey != 0)) resync_points_exact = false;
    }
    CHECK(resync_points_exact);
    CHECK(max_abs_error <= (i32)(bound * resync_interval)); // generous but genuinely bounded, not "happens to be small"

    // Lossy mode must actually compress better than lossless on the same
    // data, or there's no point to it.
    auto lossless_blob = compress_geo2d(path);
    auto lossy_blob = compress_geo2d_lossy(path, quant_step, resync_interval);
    CHECK(lossy_blob.size() < lossless_blob.size());

    // Decompress via the ordinary decompress_geo2d (no special "lossy
    // mode" API) -- quant_step/resync ride in the blob itself.
    // compress_geo2d_lossy searches candidate lags independently and may
    // not pick lag=1, so compare its result against the *original* path's
    // error bound (the same property just checked on `back`), not
    // byte-for-byte against `back` (which was computed with lag forced
    // to 1 and may legitimately differ).
    auto lossy_decoded = decompress_geo2d(lossy_blob);
    CHECK(lossy_decoded.size() == path.size());
    i32 lossy_max_abs_error = 0;
    for (size_t i = 0; i < path.size(); i++) {
        lossy_max_abs_error = std::max({lossy_max_abs_error,
                                         std::abs(path[i].x - lossy_decoded[i].x),
                                         std::abs(path[i].y - lossy_decoded[i].y)});
    }
    CHECK(lossy_max_abs_error <= (i32)(bound * resync_interval));

    // quant_step <= 1 must be exactly lossless (no approximation at all).
    auto exact_blob = compress_geo2d_lossy(path, 1, 0);
    auto exact_back = decompress_geo2d(exact_blob);
    CHECK(exact_back.size() == path.size());
    bool exact_match = true;
    for (size_t i = 0; i < path.size(); i++)
        if (exact_back[i].x != path[i].x || exact_back[i].y != path[i].y) exact_match = false;
    CHECK(exact_match);
}

// Same properties as test_rod_joint_2d_lossy, checked on the 3D true
// similarity joint: bounded per-rod error away from resync points, exact
// reset at resync points, real compression gain, and quant_step<=1 exact.
static void test_rod_joint_3d_lossy() {
    std::vector<Point3i> path;
    double x = 0, y = 0, z = 0, heading = 0;
    unsigned int seed = 314159;
    auto rnd = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    for (int i = 0; i < 3000; i++) {
        heading += (rnd() - 0.5) * 0.08;
        double speed = 8.0 + (rnd() - 0.5);
        x += speed * std::cos(heading);
        y += speed * std::sin(heading);
        z += 3.0 + (rnd() - 0.5) * 0.5; // steady climb with noise
        path.push_back({(i32)std::lround(x), (i32)std::lround(y), (i32)std::lround(z)});
    }

    u32 quant_step = 20;
    u32 resync_interval = 64;
    RodJoint3DSimResult r = rod_joint_3d_similarity_forward(path, /*lag=*/1, quant_step, resync_interval);
    auto back = rod_joint_3d_similarity_inverse(r);
    CHECK(back.size() == path.size());
    u32 bound = rod_joint_3d_error_bound(quant_step);

    bool nonresync_rods_within_bound = true;
    for (size_t i = 1; i < path.size(); i++) {
        if ((i % resync_interval) == 0) continue;
        i32 dx = std::abs((path[i].x - path[i-1].x) - (back[i].x - back[i-1].x));
        i32 dy = std::abs((path[i].y - path[i-1].y) - (back[i].y - back[i-1].y));
        i32 dz = std::abs((path[i].z - path[i-1].z) - (back[i].z - back[i-1].z));
        if ((u32)dx > bound + 1 || (u32)dy > bound + 1 || (u32)dz > bound + 1) nonresync_rods_within_bound = false;
    }
    CHECK(nonresync_rods_within_bound);

    bool resync_points_exact = true;
    i32 max_abs_error = 0;
    for (size_t i = 0; i < path.size(); i++) {
        i32 ex = std::abs(path[i].x - back[i].x), ey = std::abs(path[i].y - back[i].y), ez = std::abs(path[i].z - back[i].z);
        max_abs_error = std::max({max_abs_error, ex, ey, ez});
        if (i > 0 && (i % resync_interval) == 0 && (ex != 0 || ey != 0 || ez != 0)) resync_points_exact = false;
    }
    CHECK(resync_points_exact);
    CHECK(max_abs_error <= (i32)(bound * resync_interval));

    auto lossless_blob = compress_geo3d(path);
    auto lossy_blob = compress_geo3d_lossy(path, quant_step, resync_interval);
    CHECK(lossy_blob.size() < lossless_blob.size());

    auto lossy_decoded = decompress_geo3d(lossy_blob);
    CHECK(lossy_decoded.size() == path.size());
    i32 lossy_max_abs_error = 0;
    for (size_t i = 0; i < path.size(); i++) {
        lossy_max_abs_error = std::max({lossy_max_abs_error,
                                         std::abs(path[i].x - lossy_decoded[i].x),
                                         std::abs(path[i].y - lossy_decoded[i].y),
                                         std::abs(path[i].z - lossy_decoded[i].z)});
    }
    CHECK(lossy_max_abs_error <= (i32)(bound * resync_interval));

    auto exact_blob = compress_geo3d_lossy(path, 1, 0);
    auto exact_back = decompress_geo3d(exact_blob);
    CHECK(exact_back.size() == path.size());
    bool exact_match = true;
    for (size_t i = 0; i < path.size(); i++)
        if (exact_back[i].x != path[i].x || exact_back[i].y != path[i].y || exact_back[i].z != path[i].z)
            exact_match = false;
    CHECK(exact_match);
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

// Regression guard for the lag-search fix: a toroidal-style path (radius
// oscillating roughly every 3 rods) is a case a lag=1-only predictor
// handles poorly no matter how good its rotation/scale fit is -- the real
// fix was searching a small set of candidate lags and keeping whichever
// actually encodes smallest. This pins down that the search keeps working,
// not just that toroidal-like data round-trips.
static void test_geo3d_lag_search_toroidal() {
    std::vector<Point3i> toroidal;
    double R = 800.0, r = 250.0;
    for (int i = 0; i < 3000; i++) {
        double u = i * 0.31;
        double v = i * 2.05;
        double x = (R + r * std::cos(v)) * std::cos(u);
        double y = (R + r * std::cos(v)) * std::sin(u);
        double z = r * std::sin(v);
        toroidal.push_back({(i32)std::lround(x * 1000), (i32)std::lround(y * 1000), (i32)std::lround(z * 1000)});
    }

    auto blob = compress_geo3d(toroidal);
    auto back = decompress_geo3d(blob);
    CHECK(back.size() == toroidal.size());
    bool match = true;
    for (size_t i = 0; i < toroidal.size(); i++)
        if (back[i].x != toroidal[i].x || back[i].y != toroidal[i].y || back[i].z != toroidal[i].z)
            match = false;
    CHECK(match);

    // With lag=1 forced, this shape only compresses to roughly 85% of the
    // packed-integer raw size (an honest loss against general compressors,
    // as originally measured). The lag search should do meaningfully
    // better -- comfortably under 75% -- or this regressed back to lag=1
    // behavior in practice.
    size_t raw_packed = toroidal.size() * 3 * sizeof(i32);
    CHECK(blob.size() < raw_packed * 3 / 4);
}

// Demonstrates the actual gap per-block lag search closes: a single global
// lag (the old design -- one forward() call per candidate, whole file)
// can only ever be a compromise when the oscillation period itself
// changes partway through the sequence, since no single lag fits both
// halves. Per-block search (the new default) lets each calibration block
// pick its own best-fitting lag independently, so it should track *both*
// halves well simultaneously -- something no single forced lag can do.
static void test_rod_joint_lag_search_period_drift() {
    std::vector<Point3i> path;
    double R = 800.0, r = 250.0;
    for (int i = 0; i < 3000; i++) {
        double u = i * 0.31;
        // First half oscillates at one period, second half at a distinctly
        // different one -- no single lag predicts both well.
        double v_step = (i < 1500) ? 2.05 : 1.3;
        double v = i * v_step;
        double x = (R + r * std::cos(v)) * std::cos(u);
        double y = (R + r * std::cos(v)) * std::sin(u);
        double z = r * std::sin(v);
        path.push_back({(i32)std::lround(x * 1000), (i32)std::lround(y * 1000), (i32)std::lround(z * 1000)});
    }

    // Best possible *single* forced lag across the whole file (mirrors
    // what the old whole-file-search design would have produced), scored
    // by total residual magnitude -- the same proxy compress_geo3d's old
    // per-candidate loop implicitly optimized via actual encoded size.
    i64 best_forced_sum = -1;
    for (u32 lag : kRodJointCandidateLags) {
        RodJoint3DSimResult forced = rod_joint_3d_similarity_forward(path, lag);
        i64 sum = 0;
        for (i32 v : forced.residual_x) sum += std::abs(v);
        for (i32 v : forced.residual_y) sum += std::abs(v);
        for (i32 v : forced.residual_z) sum += std::abs(v);
        if (best_forced_sum < 0 || sum < best_forced_sum) best_forced_sum = sum;
    }

    // Per-block auto search (force_lag == 0, the default).
    RodJoint3DSimResult auto_r = rod_joint_3d_similarity_forward(path);
    i64 auto_sum = 0;
    for (i32 v : auto_r.residual_x) auto_sum += std::abs(v);
    for (i32 v : auto_r.residual_y) auto_sum += std::abs(v);
    for (i32 v : auto_r.residual_z) auto_sum += std::abs(v);

    // Round-trip correctness first -- compression quality is meaningless
    // if this doesn't hold.
    auto back = rod_joint_3d_similarity_inverse(auto_r);
    CHECK(back.size() == path.size());
    bool match = true;
    for (size_t i = 0; i < path.size(); i++)
        if (back[i].x != path[i].x || back[i].y != path[i].y || back[i].z != path[i].z) match = false;
    CHECK(match);

    // The actual claim: per-block search should meaningfully beat the best
    // *single* lag choice on a shape specifically designed so no single
    // lag fits the whole file.
    CHECK(auto_sum < best_forced_sum * 3 / 4);
    std::printf("  (lag search: best single-lag residual sum=%lld, per-block auto=%lld, %.1f%% smaller)\n",
                 (long long)best_forced_sum, (long long)auto_sum,
                 100.0 * (1.0 - (double)auto_sum / (double)best_forced_sum));
}

// Mirrors codec.cpp's serialize_geo3d_sim/serialize_quat_joint's parameter+
// residual payload (minus the small fixed header fields, which are
// roughly equal either way) closely enough to fairly compare fixed-block
// vs. adaptive-block real compressed size -- block_len contributes zero
// bytes when empty (the fixed-block case), and its real cost when
// non-empty (the adaptive case), exactly like production would pay.
static size_t measure_rodjoint3dsim_bytes(const RodJoint3DSimResult& r) {
    std::vector<u8> flat;
    for (u32 v : r.block_len) write_varint(flat, v);
    for (u32 v : r.block_lag) write_varint(flat, v);
    for (const auto& m : r.block_matrix)
        for (i64 v : m) write_varint(flat, zigzag_encode64(v));
    for (i32 v : r.residual_x) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_y) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_z) write_varint(flat, zigzag_encode32(v));
    return range_encode_bytes(flat).size();
}

static size_t measure_quatjoint_bytes(const QuaternionJointResult& r) {
    std::vector<u8> flat;
    for (u32 v : r.block_len) write_varint(flat, v);
    for (u32 v : r.block_lag) write_varint(flat, v);
    for (const auto& d : r.block_delta)
        for (i64 v : d) write_varint(flat, zigzag_encode64(v));
    for (i32 v : r.residual_w) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_x) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_y) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_z) write_varint(flat, zigzag_encode32(v));
    return range_encode_bytes(flat).size();
}

// A path deliberately alternating a very smooth, easily-predicted stretch
// (constant rotation+scale, exactly what the 3x3 joint models) with a
// jittery, hard-to-predict burst (large random noise), repeated several
// times -- so any single *fixed* block size straddles regime transitions
// somewhere. This is the position analogue of
// test_rod_joint_lag_search_period_drift, but targeting block
// *resolution* instead of *lag*: adaptive_partition should keep blocks
// large through the smooth stretches (few parameters needed) and shrink
// them through the jittery bursts (fresh calibration needed more often),
// which a fixed size can't do simultaneously.
static void test_rod_joint_3d_adaptive_blocking() {
    std::mt19937 rng(2032);
    std::uniform_int_distribution<int> jitter(-40, 40);
    std::vector<Point3i> path;
    double x = 200, y = 0, z = 0;
    double dx = 6, dy = 4, dz = 1;
    double ax = 1.0 / std::sqrt(3.0), ay = ax, az = ax;
    double theta = 0.04, ct = std::cos(theta), st = std::sin(theta);
    for (int rep = 0; rep < 8; rep++) {
        for (int i = 0; i < 300; i++) {
            path.push_back({(i32)std::lround(x), (i32)std::lround(y), (i32)std::lround(z)});
            double dot = dx * ax + dy * ay + dz * az;
            double cx = ay * dz - az * dy, cy = az * dx - ax * dz, cz = ax * dy - ay * dx;
            double ndx = dx * ct + cx * st + ax * dot * (1 - ct);
            double ndy = dy * ct + cy * st + ay * dot * (1 - ct);
            double ndz = dz * ct + cz * st + az * dot * (1 - ct);
            dx = ndx; dy = ndy; dz = ndz;
            x += dx; y += dy; z += dz;
        }
        for (int i = 0; i < 60; i++) {
            x += dx + jitter(rng); y += dy + jitter(rng); z += dz + jitter(rng);
            path.push_back({(i32)std::lround(x), (i32)std::lround(y), (i32)std::lround(z)});
        }
    }

    RodJoint3DSimResult fixed_r = rod_joint_3d_similarity_forward(path);
    RodJoint3DSimResult adaptive_r = rod_joint_3d_similarity_forward_adaptive(path);

    auto back = rod_joint_3d_similarity_inverse(adaptive_r);
    CHECK(back.size() == path.size());
    bool match = true;
    for (size_t i = 0; i < path.size(); i++)
        if (back[i].x != path[i].x || back[i].y != path[i].y || back[i].z != path[i].z) match = false;
    CHECK(match);

    size_t fixed_bytes = measure_rodjoint3dsim_bytes(fixed_r);
    size_t adaptive_bytes = measure_rodjoint3dsim_bytes(adaptive_r);
    CHECK(adaptive_bytes < fixed_bytes);
    std::printf("  (Rod-Joint 3D adaptive blocking: fixed=%zu blocks/%zu bytes, adaptive=%zu blocks/%zu bytes, %.1f%% smaller)\n",
                 fixed_r.block_lag.size(), fixed_bytes, adaptive_r.block_lag.size(), adaptive_bytes,
                 100.0 * (1.0 - (double)adaptive_bytes / (double)fixed_bytes));
}

// Orientation analogue: alternate constant-angular-velocity stretches
// with jittery bursts in the delta rotation.

namespace {
constexpr double kTestPi = 3.14159265358979323846;

// Hamilton product, unscaled doubles.
void quat_mul_d(double aw, double ax, double ay, double az, double bw, double bx, double by, double bz,
                 double& ow, double& ox, double& oy, double& oz) {
    ow = aw * bw - ax * bx - ay * by - az * bz;
    ox = aw * bx + ax * bw + ay * bz - az * by;
    oy = aw * by - ax * bz + ay * bw + az * bx;
    oz = aw * bz + ax * by - ay * bx + az * bw;
}

Quat4i quat_to_i32(double w, double x, double y, double z, double scale) {
    return {(i32)std::lround(w * scale), (i32)std::lround(x * scale), (i32)std::lround(y * scale), (i32)std::lround(z * scale)};
}
} // namespace

static void test_quaternion_joint() {
    const double scale = 1 << 20; // ~6 decimal digits of precision, same order as the GPS-track benchmark's --scale

    // Case 1: constant angular velocity about a fixed axis -- a smoothly
    // panning head or a drone holding a steady turn rate. This is the
    // shape class the lag=1 body-frame model is exactly built for: the
    // per-step delta rotation is the same every step, so a single
    // calibrated D should predict it almost perfectly.
    {
        std::vector<Quat4i> quats;
        double qw = 1, qx = 0, qy = 0, qz = 0;
        double half = (1.5 * kTestPi / 180.0) / 2.0; // 1.5 degrees/step about z
        double dqw = std::cos(half), dqz = std::sin(half);
        for (int i = 0; i < 2000; i++) {
            quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
            double nw, nx, ny, nz;
            quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
            qw = nw; qx = nx; qy = ny; qz = nz;
        }

        QuaternionJointResult r = quaternion_joint_forward(quats);
        auto back = quaternion_joint_inverse(r);
        CHECK(back.size() == quats.size());
        bool match = true;
        for (size_t i = 0; i < quats.size(); i++)
            if (back[i].w != quats[i].w || back[i].x != quats[i].x ||
                back[i].y != quats[i].y || back[i].z != quats[i].z) match = false;
        CHECK(match);

        // Real compression claim: constant angular velocity should predict
        // almost exactly, so residuals should be tiny compared to the raw
        // quaternion component magnitudes (~scale).
        i64 res_sum = 0;
        for (i32 v : r.residual_w) res_sum += std::abs(v);
        for (i32 v : r.residual_x) res_sum += std::abs(v);
        for (i32 v : r.residual_y) res_sum += std::abs(v);
        for (i32 v : r.residual_z) res_sum += std::abs(v);
        double avg_res = (double)res_sum / (double)(4 * (quats.size() - 1));
        CHECK(avg_res < scale * 0.001); // residuals under ~0.1% of a unit-quaternion component
        std::printf("  (Quaternion Joint, constant angular velocity: avg |residual|=%.2f, scale=%.0f)\n", avg_res, scale);
    }

    // Case 2: oscillating angular velocity (a back-and-forth head shake,
    // period 6 steps) with two distinct periods in the first vs second
    // half -- the same "no single lag fits the whole file" design as
    // test_rod_joint_lag_search_period_drift, to check the lag search
    // benefit carries over to orientation.
    {
        std::vector<Quat4i> quats;
        double qw = 1, qx = 0, qy = 0, qz = 0;
        for (int i = 0; i < 3000; i++) {
            quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
            double period = (i < 1500) ? 6.0 : 4.0;
            double amp_deg = 2.0 * std::sin(2.0 * kTestPi * (double)i / period);
            double half = (amp_deg * kTestPi / 180.0) / 2.0;
            double dqw = std::cos(half), dqx = std::sin(half);
            double nw, nx, ny, nz;
            quat_mul_d(qw, qx, qy, qz, dqw, dqx, 0, 0, nw, nx, ny, nz);
            qw = nw; qx = nx; qy = ny; qz = nz;
        }

        i64 best_forced_sum = -1;
        for (u32 lag : kRodJointCandidateLags) {
            QuaternionJointResult forced = quaternion_joint_forward(quats, lag);
            i64 sum = 0;
            for (i32 v : forced.residual_w) sum += std::abs(v);
            for (i32 v : forced.residual_x) sum += std::abs(v);
            for (i32 v : forced.residual_y) sum += std::abs(v);
            for (i32 v : forced.residual_z) sum += std::abs(v);
            if (best_forced_sum < 0 || sum < best_forced_sum) best_forced_sum = sum;
        }

        QuaternionJointResult auto_r = quaternion_joint_forward(quats);
        i64 auto_sum = 0;
        for (i32 v : auto_r.residual_w) auto_sum += std::abs(v);
        for (i32 v : auto_r.residual_x) auto_sum += std::abs(v);
        for (i32 v : auto_r.residual_y) auto_sum += std::abs(v);
        for (i32 v : auto_r.residual_z) auto_sum += std::abs(v);

        auto back = quaternion_joint_inverse(auto_r);
        CHECK(back.size() == quats.size());
        bool match = true;
        for (size_t i = 0; i < quats.size(); i++)
            if (back[i].w != quats[i].w || back[i].x != quats[i].x ||
                back[i].y != quats[i].y || back[i].z != quats[i].z) match = false;
        CHECK(match);

        CHECK(auto_sum < best_forced_sum * 3 / 4);
        std::printf("  (Quaternion Joint lag search: best single-lag residual sum=%lld, per-block auto=%lld, %.1f%% smaller)\n",
                     (long long)best_forced_sum, (long long)auto_sum,
                     100.0 * (1.0 - (double)auto_sum / (double)best_forced_sum));
    }

    // Edge cases.
    {
        std::vector<Quat4i> empty;
        CHECK(quaternion_joint_inverse(quaternion_joint_forward(empty)).empty());

        std::vector<Quat4i> one = {{(i32)scale, 0, 0, 0}};
        auto back1 = quaternion_joint_inverse(quaternion_joint_forward(one));
        CHECK(back1.size() == 1 && back1[0].w == one[0].w);

        std::vector<Quat4i> two = {{(i32)scale, 0, 0, 0}, {0, (i32)scale, 0, 0}};
        auto back2 = quaternion_joint_inverse(quaternion_joint_forward(two));
        CHECK(back2.size() == 2 && back2[1].x == two[1].x);
    }

    // Adversarial case: random, structurally inconsistent quaternion-shaped
    // integers (not even normalized) -- the calibrated model should predict
    // poorly (large residuals expected), but round-trip must still be exact
    // regardless, exactly like rod_joint_2d's random_walk case.
    {
        std::mt19937 rng(2029);
        std::uniform_int_distribution<int> d(-(1 << 20), 1 << 20);
        std::vector<Quat4i> randq;
        for (int i = 0; i < 500; i++) randq.push_back({d(rng), d(rng), d(rng), d(rng)});
        auto back = quaternion_joint_inverse(quaternion_joint_forward(randq));
        CHECK(back.size() == randq.size());
        bool match = true;
        for (size_t i = 0; i < randq.size(); i++)
            if (back[i].w != randq[i].w || back[i].x != randq[i].x ||
                back[i].y != randq[i].y || back[i].z != randq[i].z) match = false;
        CHECK(match);
    }
}

static void test_quaternion_joint_lossy() {
    const double scale = 1 << 20;
    std::vector<Quat4i> quats;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    unsigned int seed = 2030;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    for (int i = 0; i < 4000; i++) {
        quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
        double deg = 1.0 + (next_rand() - 0.5) * 0.4; // steady turn rate with small noise
        double half = (deg * kTestPi / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw, nx, ny, nz;
        quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    // quant_step <= 1 is exactly lossless.
    {
        QuaternionJointResult r0 = quaternion_joint_forward(quats, 0, 0, 0);
        QuaternionJointResult r1 = quaternion_joint_forward(quats, 0, 1, 0);
        auto back0 = quaternion_joint_inverse(r0);
        auto back1 = quaternion_joint_inverse(r1);
        bool match = true;
        for (size_t i = 0; i < quats.size(); i++) {
            if (back0[i].w != quats[i].w || back0[i].x != quats[i].x || back0[i].y != quats[i].y || back0[i].z != quats[i].z) match = false;
            if (back1[i].w != quats[i].w || back1[i].x != quats[i].x || back1[i].y != quats[i].y || back1[i].z != quats[i].z) match = false;
        }
        CHECK(match);
    }

    // Lossy: bounded error, periodic resync limits drift, real compression win.
    u32 quant_step = 64, resync_interval = 32;
    QuaternionJointResult lossless = quaternion_joint_forward(quats, 0, 1, 0);
    QuaternionJointResult lossy = quaternion_joint_forward(quats, 0, quant_step, resync_interval);
    auto back = quaternion_joint_inverse(lossy);
    CHECK(back.size() == quats.size());

    i32 max_err = 0;
    for (size_t i = 0; i < quats.size(); i++)
        max_err = std::max({max_err, std::abs(quats[i].w - back[i].w), std::abs(quats[i].x - back[i].x),
                             std::abs(quats[i].y - back[i].y), std::abs(quats[i].z - back[i].z)});
    // Generous bound: resync_interval steps between exact keyframes, each
    // step's own quantization error up to quant_step/2, plus headroom for
    // compounding through the recursive prediction dependency.
    i32 generous_bound = (i32)quant_step * (i32)resync_interval;
    CHECK(max_err <= generous_bound);

    i64 lossless_res_sum = 0, lossy_res_sum = 0;
    for (i32 v : lossless.residual_w) lossless_res_sum += std::abs(v);
    for (i32 v : lossless.residual_x) lossless_res_sum += std::abs(v);
    for (i32 v : lossless.residual_y) lossless_res_sum += std::abs(v);
    for (i32 v : lossless.residual_z) lossless_res_sum += std::abs(v);
    for (i32 v : lossy.residual_w) lossy_res_sum += std::abs(v);
    for (i32 v : lossy.residual_x) lossy_res_sum += std::abs(v);
    for (i32 v : lossy.residual_y) lossy_res_sum += std::abs(v);
    for (i32 v : lossy.residual_z) lossy_res_sum += std::abs(v);
    CHECK(lossy_res_sum < lossless_res_sum); // quantized residual indices are genuinely smaller in magnitude
    std::printf("  (Quaternion Joint lossy: quant_step=%u, max observed error=%d (bound %d), residual magnitude sum %lld -> %lld)\n",
                 quant_step, max_err, generous_bound, (long long)lossless_res_sum, (long long)lossy_res_sum);
}

// Orientation analogue of test_rod_joint_3d_adaptive_blocking: alternate
// constant-angular-velocity stretches with jittery bursts in the delta
// rotation.
static void test_quaternion_joint_adaptive_blocking() {
    std::mt19937 rng(2033);
    std::uniform_real_distribution<double> jitter_deg(-4.0, 4.0);
    std::vector<Quat4i> quats;
    const double scale = 1 << 20;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    for (int rep = 0; rep < 8; rep++) {
        for (int i = 0; i < 300; i++) {
            quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
            double half = (1.5 * kTestPi / 180.0) / 2.0;
            double dqw = std::cos(half), dqz = std::sin(half);
            double nw, nx, ny, nz;
            quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
        for (int i = 0; i < 60; i++) {
            quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
            double half = (jitter_deg(rng) * kTestPi / 180.0) / 2.0;
            double dqw = std::cos(half), dqz = std::sin(half);
            double nw, nx, ny, nz;
            quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
            qw = nw; qx = nx; qy = ny; qz = nz;
        }
    }

    QuaternionJointResult fixed_r = quaternion_joint_forward(quats);
    QuaternionJointResult adaptive_r = quaternion_joint_forward_adaptive(quats);

    auto back = quaternion_joint_inverse(adaptive_r);
    CHECK(back.size() == quats.size());
    bool match = true;
    for (size_t i = 0; i < quats.size(); i++)
        if (back[i].w != quats[i].w || back[i].x != quats[i].x ||
            back[i].y != quats[i].y || back[i].z != quats[i].z) match = false;
    CHECK(match);

    size_t fixed_bytes = measure_quatjoint_bytes(fixed_r);
    size_t adaptive_bytes = measure_quatjoint_bytes(adaptive_r);
    CHECK(adaptive_bytes < fixed_bytes);
    std::printf("  (Quaternion Joint adaptive blocking: fixed=%zu blocks/%zu bytes, adaptive=%zu blocks/%zu bytes, %.1f%% smaller)\n",
                 fixed_r.block_lag.size(), fixed_bytes, adaptive_r.block_lag.size(), adaptive_bytes,
                 100.0 * (1.0 - (double)adaptive_bytes / (double)fixed_bytes));
}

// Validates quat_calibrate_all_blocks_cuda against the CPU's own
// per-(block, forced-lag) calibration (quaternion_joint_forward's
// force_lag path) BEFORE any GPU timing claim is trusted -- matching this
// project's standing rule that a GPU kernel's output is verified against
// the CPU path first, timed second, never the other way around.
static void test_quat_calibration_cuda_matches_cpu() {
    if (!cuda_is_available()) return;

    std::mt19937 rng(5051);
    std::uniform_real_distribution<double> noise_deg(-3.0, 3.0);
    std::vector<Quat4i> quats;
    const double scale = 1 << 20;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    for (int i = 0; i < 3000; i++) {
        quats.push_back(quat_to_i32(qw, qx, qy, qz, scale));
        double half = ((1.5 + noise_deg(rng)) * kTestPi / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw, nx, ny, nz;
        quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    size_t n = quats.size() - 1;
    std::vector<i32> qw_arr(n), qx_arr(n), qy_arr(n), qz_arr(n);
    for (size_t i = 1; i < quats.size(); i++) {
        qw_arr[i - 1] = quats[i].w; qx_arr[i - 1] = quats[i].x;
        qy_arr[i - 1] = quats[i].y; qz_arr[i - 1] = quats[i].z;
    }
    std::vector<u32> lags(kRodJointCandidateLags, kRodJointCandidateLags + kRodJointNumCandidateLags);

    std::vector<std::array<i64, 4>> gpu_delta;
    std::vector<double> gpu_sse;
    bool ok = quat_calibrate_all_blocks_cuda(qw_arr, qx_arr, qy_arr, qz_arr, kQuatJointBlockSize, lags, gpu_delta, gpu_sse);
    CHECK(ok);
    if (!ok) return;

    size_t nblocks = (n + kQuatJointBlockSize - 1) / kQuatJointBlockSize;
    CHECK(gpu_delta.size() == nblocks * lags.size());

    // Cross-check every (block, lag) pair's calibrated D against the
    // CPU's own forced-lag path (quaternion_joint_forward(quats, lag)).
    // Allowing +-1 in the Q16.16 integer for floating-point summation-
    // order differences between the CPU's sequential accumulation and
    // the GPU's tree reduction -- not a claim of bit-exact reproduction,
    // the same tolerance this codebase always allows for calibration
    // math (only the final integer residual needs to be exact).
    bool all_match = true;
    int mismatches = 0;
    for (size_t lag_idx = 0; lag_idx < lags.size(); lag_idx++) {
        QuaternionJointResult cpu_r = quaternion_joint_forward(quats, lags[lag_idx]);
        CHECK(cpu_r.block_delta.size() == nblocks);
        for (size_t b = 0; b < nblocks; b++) {
            const auto& cpu_d = cpu_r.block_delta[b];
            const auto& gpu_d = gpu_delta[b * lags.size() + lag_idx];
            for (int k = 0; k < 4; k++) {
                if (std::abs(cpu_d[k] - gpu_d[k]) > 1) { all_match = false; mismatches++; }
            }
        }
    }
    CHECK(all_match);
    std::printf("  (Quaternion calibration CUDA: %zu blocks x %zu lags cross-checked against CPU, mismatches=%d)\n",
                 nblocks, lags.size(), mismatches);
}

// End-to-end through the public compress_pose/decompress_pose/
// compress_pose_lossy API: a synthetic 6-DOF pose stream (a drone
// circling upward while smoothly yawing to track its own heading, plus
// small realistic per-step noise on both position and orientation) --
// close to the real GPS-trajectory/pose-tracking shape class both halves
// of this codec target, not a hand-picked best case for either.
static void test_pose_codec() {
    const double qscale = 1 << 20;
    std::vector<Pose> poses;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    unsigned int seed = 2031;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    double z = 0.0;
    for (int i = 0; i < 1500; i++) {
        double t = i * 0.05;
        i32 px = (i32)std::lround(2000 * std::cos(t));
        i32 py = (i32)std::lround(2000 * std::sin(t));
        z += 4.0 + (next_rand() - 0.5) * 0.5;
        i32 pz = (i32)std::lround(z);
        poses.push_back({{px, py, pz}, quat_to_i32(qw, qx, qy, qz, qscale)});

        double deg = 2.0 + (next_rand() - 0.5) * 0.3; // yaw tracking the circular heading, plus noise
        double half = (deg * kTestPi / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw, nx, ny, nz;
        quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    // Lossless round trip.
    std::vector<u8> blob = compress_pose(poses);
    std::vector<Pose> back = decompress_pose(blob);
    CHECK(back.size() == poses.size());
    bool match = true;
    for (size_t i = 0; i < poses.size(); i++) {
        const Pose& a = poses[i]; const Pose& b = back[i];
        if (a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
            a.orientation.w != b.orientation.w || a.orientation.x != b.orientation.x ||
            a.orientation.y != b.orientation.y || a.orientation.z != b.orientation.z) match = false;
    }
    CHECK(match);

    // Real compression win over a fair raw-packed baseline (7 int32s per
    // pose: x,y,z,qw,qx,qy,qz), the same discipline BENCHMARKS.md uses.
    size_t raw_packed = poses.size() * 7 * sizeof(i32);
    CHECK(blob.size() < raw_packed);
    std::printf("  (Pose codec: %zu poses, raw packed~=%zu bytes -> %zu bytes, %.1f%% smaller)\n",
                 poses.size(), raw_packed, blob.size(), 100.0 * (1.0 - (double)blob.size() / (double)raw_packed));

    // Lossy: quant_step <= 1 on both halves is exactly compress_pose.
    std::vector<u8> exact_blob = compress_pose_lossy(poses, 1, 0, 1, 0);
    std::vector<Pose> exact_back = decompress_pose(exact_blob);
    bool exact_match = exact_back.size() == poses.size();
    for (size_t i = 0; exact_match && i < poses.size(); i++) {
        const Pose& a = poses[i]; const Pose& b = exact_back[i];
        if (a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
            a.orientation.w != b.orientation.w || a.orientation.x != b.orientation.x ||
            a.orientation.y != b.orientation.y || a.orientation.z != b.orientation.z) exact_match = false;
    }
    CHECK(exact_match);

    // Lossy with real quantization on both halves: bounded error, smaller
    // than lossless.
    u32 pos_q = 8, pos_resync = 64, quat_q = 32, quat_resync = 32;
    std::vector<u8> lossy_blob = compress_pose_lossy(poses, pos_q, pos_resync, quat_q, quat_resync);
    std::vector<Pose> lossy_back = decompress_pose(lossy_blob);
    CHECK(lossy_back.size() == poses.size());
    CHECK(lossy_blob.size() < blob.size());

    i32 max_pos_err = 0, max_quat_err = 0;
    for (size_t i = 0; i < poses.size(); i++) {
        const Pose& a = poses[i]; const Pose& b = lossy_back[i];
        max_pos_err = std::max({max_pos_err, std::abs(a.position.x - b.position.x),
                                 std::abs(a.position.y - b.position.y), std::abs(a.position.z - b.position.z)});
        max_quat_err = std::max({max_quat_err, std::abs(a.orientation.w - b.orientation.w),
                                  std::abs(a.orientation.x - b.orientation.x),
                                  std::abs(a.orientation.y - b.orientation.y),
                                  std::abs(a.orientation.z - b.orientation.z)});
    }
    std::printf("  (Pose codec lossy: %zu -> %zu bytes, max position error=%d, max orientation-component error=%d)\n",
                 blob.size(), lossy_blob.size(), max_pos_err, max_quat_err);
}

// Validates PoseStreamEncoder/PoseStreamDecoder: simulates a live feed by
// pushing poses one at a time (never handing the encoder the whole
// sequence at once), collecting the bytes it emits incrementally exactly
// as a real transport would deliver them, and feeding those bytes to the
// decoder in small, deliberately-misaligned pieces (not one chunk at a
// time) to prove it correctly handles partial/arbitrary delivery
// boundaries, not just "one write() per chunk." Also reports the real,
// measured ratio cost of chunking (every chunk restarts prediction from
// zero, unlike one continuous batch encode) against compress_pose on the
// identical pose sequence.
static void test_pose_stream() {
    const double qscale = 1 << 20;
    std::vector<Pose> poses;
    double qw = 1, qx = 0, qy = 0, qz = 0;
    unsigned int seed = 4242;
    auto next_rand = [&]() { seed = seed * 1664525u + 1013904223u; return (double)(seed >> 8) / (double)(1u << 24); };
    double z = 0.0;
    for (int i = 0; i < 2000; i++) {
        double t = i * 0.05;
        i32 px = (i32)std::lround(2000 * std::cos(t));
        i32 py = (i32)std::lround(2000 * std::sin(t));
        z += 4.0 + (next_rand() - 0.5) * 0.5;
        i32 pz = (i32)std::lround(z);
        poses.push_back({{px, py, pz}, quat_to_i32(qw, qx, qy, qz, qscale)});

        double deg = 2.0 + (next_rand() - 0.5) * 0.3;
        double half = (deg * kTestPi / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw, nx, ny, nz;
        quat_mul_d(qw, qx, qy, qz, dqw, 0, 0, dqz, nw, nx, ny, nz);
        qw = nw; qx = nx; qy = ny; qz = nz;
    }

    // Encode: push one at a time, collect emitted bytes as a single
    // growing buffer standing in for "what a real transport delivered."
    std::vector<u8> wire;
    size_t max_buffered_during_encode = 0;
    {
        PoseStreamEncoder enc([&](const std::vector<u8>& chunk) {
            wire.insert(wire.end(), chunk.begin(), chunk.end());
        }, /*chunk_size=*/128);
        for (const Pose& p : poses) enc.push(p);
        enc.finish();
    }
    (void)max_buffered_during_encode;
    CHECK(!wire.empty());

    // Decode: feed bytes back in small, arbitrarily-sized (and
    // deliberately NOT chunk-aligned) pieces, to prove the decoder
    // correctly buffers a partial chunk instead of assuming each feed()
    // call lines up with a chunk boundary.
    std::vector<Pose> decoded;
    {
        PoseStreamDecoder dec([&](const Pose& p) { decoded.push_back(p); });
        size_t pos = 0;
        size_t step = 37; // deliberately not aligned to any chunk/header boundary
        while (pos < wire.size()) {
            size_t n = std::min(step, wire.size() - pos);
            dec.feed(wire.data() + pos, n);
            pos += n;
        }
        CHECK(dec.finished());
    }

    CHECK(decoded.size() == poses.size());
    bool match = true;
    for (size_t i = 0; i < poses.size() && i < decoded.size(); i++) {
        const Pose& a = poses[i]; const Pose& b = decoded[i];
        if (a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
            a.orientation.w != b.orientation.w || a.orientation.x != b.orientation.x ||
            a.orientation.y != b.orientation.y || a.orientation.z != b.orientation.z) match = false;
    }
    CHECK(match);

    // Real, measured ratio cost of chunking vs. one continuous batch encode.
    std::vector<u8> batch_blob = compress_pose(poses);
    std::printf("  (Pose stream: %zu poses in %zu chunks, streamed=%zu bytes vs batch=%zu bytes (%.1f%% larger), round-trip=%s)\n",
                 poses.size(), (poses.size() + 127) / 128, wire.size(), batch_blob.size(),
                 100.0 * ((double)wire.size() / (double)batch_blob.size() - 1.0),
                 match ? "PASS" : "FAIL");
}

// Validates encode_interleaved_rans/decode_interleaved_rans round-trip
// across the same kind of edge cases test_range_coder already exercises
// (empty, single byte, constant, periodic, high-entropy, every-symbol-
// once), plus cases specific to interleaving itself (more lanes than
// bytes, a single lane, lane counts that don't evenly divide the input).
// Then measures, honestly, the two things the header comment promises to
// measure rather than assume: the real ratio cost of this order-0 static
// coder against the existing order-1 adaptive range_encode_bytes on the
// same realistic (skewed, non-random) data, and the real wall-clock
// effect of splitting that data into more lanes/threads.
static void test_rans_coder() {
    std::mt19937 rng(1234);
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
    cases.push_back(random_bytes(3, rng)); // fewer bytes than lanes

    for (auto& input : cases) {
        for (int num_lanes : {1, 4, 8}) {
            auto coded = encode_interleaved_rans(input, num_lanes);
            auto decoded = decode_interleaved_rans(coded);
            CHECK(decoded == input);
        }
    }

    // Constant, highly periodic data should still shrink substantially
    // even under a static order-0 table (every byte maps to the same,
    // highly-skewed frequency regardless of position).
    std::vector<u8> constant(10000, 42);
    auto coded_constant = encode_interleaved_rans(constant, 4);
    CHECK(coded_constant.size() < constant.size() / 10);
    CHECK(decode_interleaved_rans(coded_constant) == constant);

    // Realistic skewed (non-random, non-constant) data: a synthetic
    // English-letter-frequency-like byte stream, standing in for the
    // "most realistic data" the header comment says this coder pays a
    // real ratio cost on relative to the adaptive order-1 range coder.
    std::vector<u8> skewed;
    {
        const char* alphabet = "etaoinshrdlucmfwypvbgkjqxz";
        std::uniform_real_distribution<double> u(0.0, 1.0);
        for (int i = 0; i < 200000; i++) {
            double r = u(rng);
            int idx = (int)(26 * r * r); // quadratic skew toward the front (common letters)
            if (idx > 25) idx = 25;
            skewed.push_back((u8)alphabet[idx]);
        }
    }
    auto rans_coded = encode_interleaved_rans(skewed, 4);
    auto rans_decoded = decode_interleaved_rans(rans_coded);
    CHECK(rans_decoded == skewed);
    auto range_coded = range_encode_bytes(skewed);
    auto range_decoded = range_decode_bytes(range_coded.data(), range_coded.size(), skewed.size());
    CHECK(range_decoded == skewed);
    std::printf("  (rANS vs range coder on skewed data: rans=%zu bytes, range(order-1 adaptive)=%zu bytes (rans is %.1f%% larger))\n",
                 rans_coded.size(), range_coded.size(),
                 100.0 * ((double)rans_coded.size() / (double)range_coded.size() - 1.0));

    // Real wall-clock effect of lane count on encode/decode time for a
    // larger buffer, reported honestly (thread overhead can dominate at
    // small sizes -- this buffer is large enough that it shouldn't, but
    // the number printed is what actually happened, not an assumption).
    std::vector<u8> big = random_bytes(0, rng);
    {
        std::vector<u8> v;
        for (int i = 0; i < 4000000; i++) v.push_back((u8)((i * 2654435761u) % 256));
        big = v;
    }
    std::vector<u8> reference_coded;
    for (int num_lanes : {1, 4, 8}) {
        auto t0 = std::chrono::steady_clock::now();
        auto coded_big = encode_interleaved_rans(big, num_lanes);
        auto t1 = std::chrono::steady_clock::now();
        auto decoded_big = decode_interleaved_rans(coded_big);
        auto t2 = std::chrono::steady_clock::now();
        CHECK(decoded_big == big);
        // Deterministic parallelism (see docs/PARALLELISM.md): re-running
        // the exact same lane count must produce byte-identical output --
        // the thread pool's task scheduling order must never leak into
        // the result, since each lane writes to its own disjoint output
        // range regardless of which worker thread or in what order it runs.
        auto coded_big_rerun = encode_interleaved_rans(big, num_lanes);
        CHECK(coded_big_rerun == coded_big);
        double enc_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double dec_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("  (rANS %d lane(s), %zu bytes: encode=%.2fms decode=%.2fms)\n",
                     num_lanes, big.size(), enc_ms, dec_ms);
    }
}

// The thread pool is a general-purpose primitive, not something only
// rANS-shaped code should trust -- tested directly here rather than only
// indirectly through test_rans_coder() above.
static void test_thread_pool() {
    {
        ThreadPool pool(4);
        CHECK(pool.size() == 4);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 100; i++) futures.push_back(pool.submit([i] { return i * i; }));
        bool all_correct = true;
        for (int i = 0; i < 100; i++) if (futures[(size_t)i].get() != i * i) all_correct = false;
        CHECK(all_correct);
    } // pool destructor runs here: must join every worker, never hang, never leak a thread

    // Exceptions thrown inside a submitted task must surface at the
    // future, not crash the worker thread or the process.
    {
        ThreadPool pool(2);
        auto fut = pool.submit([]() -> int { throw std::runtime_error("intentional test exception"); });
        bool threw = false;
        try { fut.get(); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "intentional test exception"; }
        CHECK(threw);
        // the pool must still be usable after a task throws
        auto fut2 = pool.submit([] { return 42; });
        CHECK(fut2.get() == 42);
    }

    // parallel_for's fn(0..count-1) contract, and its default pool being
    // real and shared (default_thread_pool() returns the same instance
    // across calls -- checked indirectly by confirming a real speedup is
    // possible without constructing a new pool per call).
    {
        std::vector<int> seen(16, -1);
        parallel_for(16, [&](int i) { seen[(size_t)i] = i * 2; });
        bool all_correct = true;
        for (int i = 0; i < 16; i++) if (seen[(size_t)i] != i * 2) all_correct = false;
        CHECK(all_correct);

        // count<=0 must be a safe no-op, not undefined behavior.
        int calls = 0;
        parallel_for(0, [&](int) { calls++; });
        CHECK(calls == 0);

        // An exception from any single index must propagate to the caller.
        bool threw = false;
        try {
            parallel_for(8, [](int i) { if (i == 3) throw std::runtime_error("boom"); });
        } catch (const std::runtime_error&) { threw = true; }
        CHECK(threw);

        CHECK(&default_thread_pool() == &default_thread_pool()); // same instance every call
    }
}

// max_abs_diff_i32's AVX2 kernel is only meaningful to compare against
// the scalar reference on hardware that actually has AVX2 -- detected at
// runtime (csa::detect_simd_backend()), not assumed. On hardware without
// it, this test still exercises and trusts the scalar path (the only
// one that ran at all for csa_capi/every real caller on that machine).
static void test_simd_max_abs_diff() {
    std::mt19937 rng(20240615);
    std::uniform_int_distribution<i32> dist(-1000000000, 1000000000);
    auto rnd_i32 = [&](i32 /*lo*/, i32 /*hi*/) -> i32 { return dist(rng); };

    for (size_t n : {size_t(0), size_t(1), size_t(3), size_t(7), size_t(8), size_t(9), size_t(1000), size_t(100003)}) {
        std::vector<i32> a(n), b(n);
        for (size_t i = 0; i < n; i++) {
            // Bounded well within int32 so a[i]-b[i] can never overflow --
            // matching max_abs_diff_i32_avx2's documented precondition
            // (see src/simd_avx2.cpp), not testing past it.
            a[i] = rnd_i32(-1000000000, 1000000000);
            b[i] = rnd_i32(-1000000000, 1000000000);
        }
        u32 scalar_result = max_abs_diff_i32_scalar(a.data(), b.data(), n);
        u32 dispatched_result = max_abs_diff_i32(a.data(), b.data(), n);
        CHECK(dispatched_result == scalar_result);

#if CSA_X86_SIMD
        if (detect_simd_backend() == SimdBackend::AVX2) {
            u32 avx2_result = max_abs_diff_i32_avx2(a.data(), b.data(), n);
            CHECK(avx2_result == scalar_result);
        }
#endif
    }

    // A real, order-independent case: the max must come from whichever
    // element actually has it, regardless of where that element falls
    // relative to AVX2's 8-wide lane boundaries (once at the very start,
    // once crossing a lane boundary, once in the scalar tail).
    {
        std::vector<i32> a(19, 0), b(19, 0);
        a[0] = 1000000000; b[0] = -1000000000;   // lane 0, element 0
        a[9] = 500000000; b[9] = -500000000;      // crosses into the second 8-wide lane
        a[18] = 300000000; b[18] = 0;             // scalar tail (19 % 8 == 3)
        CHECK(max_abs_diff_i32(a.data(), b.data(), 19) == 2000000000u);
    }

    std::printf("  (SIMD backend detected: %s)\n", detect_simd_backend() == SimdBackend::AVX2 ? "AVX2" : "scalar");

    // Microbenchmark, not an end-to-end compression claim: this isolates
    // exactly the kernel this pass vectorized, on a large buffer, so the
    // reported speedup is attributable to this specific change and
    // nothing else running at the same time.
#if CSA_X86_SIMD
    if (detect_simd_backend() == SimdBackend::AVX2) {
        size_t n = 20000000;
        std::vector<i32> a(n), b(n);
        for (size_t i = 0; i < n; i++) { a[i] = (i32)(i % 1000000); b[i] = (i32)((i * 7) % 1000000); }

        auto t0 = std::chrono::steady_clock::now();
        volatile u32 r1 = max_abs_diff_i32_scalar(a.data(), b.data(), n);
        auto t1 = std::chrono::steady_clock::now();
        volatile u32 r2 = max_abs_diff_i32_avx2(a.data(), b.data(), n);
        auto t2 = std::chrono::steady_clock::now();
        CHECK(r1 == r2);

        double scalar_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double avx2_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::printf("  (max_abs_diff_i32: scalar=%.2fms AVX2=%.2fms on %zu elements, %.2fx speedup)\n",
                     scalar_ms, avx2_ms, n, scalar_ms / avx2_ms);
    } else {
        std::printf("  (AVX2 not available on this CPU -- microbenchmark skipped, scalar path already verified above)\n");
    }
#else
    std::printf("  (non-x86 build -- no AVX2 kernel exists on this target; scalar path already verified above)\n");
#endif
}

// The third backend for the same reduction test_simd_max_abs_diff()
// above already checks two of (scalar CPU, AVX2, GPU): correctness
// against the scalar reference across several sizes, then a real,
// honestly-reported timing comparison against the best available CPU
// path at a large size -- including the H2D/D2H transfer cost in the
// GPU number, since that's what a real caller actually pays, not just
// the kernel's own execution time.
static void test_simd_max_abs_diff_cuda() {
    if (!cuda_is_available()) {
        std::printf("  (CUDA not available at test time; GPU SIMD path skipped, CPU-only verified above)\n");
        return;
    }

    std::mt19937 rng(20240616);
    std::uniform_int_distribution<i32> dist(-1000000000, 1000000000);
    for (size_t n : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(9), size_t(1000), size_t(100003)}) {
        std::vector<i32> a(n), b(n);
        for (size_t i = 0; i < n; i++) { a[i] = dist(rng); b[i] = dist(rng); }
        u32 cpu_result = max_abs_diff_i32_scalar(a.data(), b.data(), n);
        u32 gpu_result = 0;
        bool ok = max_abs_diff_i32_cuda(a.data(), b.data(), n, gpu_result);
        CHECK(ok);
        CHECK(gpu_result == cpu_result);
    }

    size_t n = 20000000;
    std::vector<i32> a(n), b(n);
    for (size_t i = 0; i < n; i++) { a[i] = (i32)(i % 1000000); b[i] = (i32)((i * 7) % 1000000); }

    auto t0 = std::chrono::steady_clock::now();
    volatile u32 r1 = max_abs_diff_i32(a.data(), b.data(), n); // best available CPU backend
    auto t1 = std::chrono::steady_clock::now();
    u32 r2 = 0;
    bool ok = max_abs_diff_i32_cuda(a.data(), b.data(), n, r2);
    auto t2 = std::chrono::steady_clock::now();
    CHECK(ok);
    CHECK(r1 == r2);

    double cpu_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double gpu_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    std::printf("  (max_abs_diff_i32 CUDA: CPU(best)=%.2fms GPU(incl. transfer)=%.2fms on %zu elements, %.2fx)\n",
                 cpu_ms, gpu_ms, n, cpu_ms / gpu_ms);
}

// Deliberately spends real wall-clock time (not just increments a
// counter) so a timing comparison against it is measuring real work,
// not the cost of scheduling an empty no-op.
static void busy_work(int iterations) {
    volatile double acc = 0.0;
    for (int i = 0; i < iterations; i++) acc += std::sin((double)i) * std::cos((double)i);
}

static void test_work_stealing_pool() {
    // Correctness: every submitted task actually runs, with the right
    // result, exactly once.
    {
        WorkStealingPool pool(4);
        CHECK(pool.size() == 4);
        std::vector<std::future<int>> futures;
        for (int i = 0; i < 200; i++) futures.push_back(pool.submit([i] { return i + 1; }));
        bool all_correct = true;
        for (int i = 0; i < 200; i++) if (futures[(size_t)i].get() != i + 1) all_correct = false;
        CHECK(all_correct);
        WorkStealingStats s = pool.stats();
        CHECK(s.tasks_submitted == 200);
        CHECK(s.tasks_completed == 200);
    } // destructor must join every worker without hanging

    // Exceptions propagate through the future, don't crash a worker.
    {
        WorkStealingPool pool(2);
        auto fut = pool.submit([]() -> int { throw std::runtime_error("intentional work-stealing test exception"); });
        bool threw = false;
        try { fut.get(); } catch (const std::runtime_error& e) { threw = std::string(e.what()) == "intentional work-stealing test exception"; }
        CHECK(threw);
        CHECK(pool.submit([] { return 7; }).get() == 7); // pool still usable afterward
    }

    // The real point of this class: a genuinely imbalanced workload,
    // where a handful of tasks cost far more than the rest, submitted in
    // round-robin order (so a naive *static* split -- assign task i to
    // worker i%N, no stealing at all -- concentrates all the expensive
    // tasks on one worker while the others sit idle). Compares actual
    // wall-clock time for that static baseline against the same task set
    // run through WorkStealingPool, which can rebalance via stealing.
    // This is a measured comparison, not an assumption that stealing
    // "must" help -- if it didn't, that would be a real, reportable
    // finding too.
    {
        const int num_workers = 4;
        const int num_tasks = 64;
        const int heavy_every = num_workers; // every heavy_every-th task (by submission order) is expensive
        const int heavy_iters = 300000, light_iters = 3000;

        // Baseline: statically partition tasks round-robin across
        // num_workers real std::threads, each running its assigned
        // tasks strictly in order on its own thread -- no stealing, no
        // shared queue, the simplest possible parallel baseline.
        auto run_static_baseline = [&]() -> double {
            std::vector<std::vector<int>> assignment(num_workers);
            for (int i = 0; i < num_tasks; i++) assignment[i % num_workers].push_back(i);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::thread> threads;
            for (int w = 0; w < num_workers; w++) {
                threads.emplace_back([&assignment, w, heavy_every, heavy_iters, light_iters] {
                    for (int task_id : assignment[(size_t)w]) busy_work(task_id % heavy_every == 0 ? heavy_iters : light_iters);
                });
            }
            for (auto& t : threads) t.join();
            auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        auto run_work_stealing = [&]() -> double {
            WorkStealingPool pool((size_t)num_workers);
            auto t0 = std::chrono::steady_clock::now();
            std::vector<std::future<void>> futures;
            for (int i = 0; i < num_tasks; i++) {
                int iters = (i % heavy_every == 0) ? heavy_iters : light_iters;
                futures.push_back(pool.submit([iters] { busy_work(iters); }));
            }
            for (auto& f : futures) f.get();
            auto t1 = std::chrono::steady_clock::now();
            return std::chrono::duration<double, std::milli>(t1 - t0).count();
        };

        double static_ms = run_static_baseline();
        double stealing_ms = run_work_stealing();
        std::printf("  (work-stealing vs static split, imbalanced workload: static=%.2fms work-stealing=%.2fms, %.2fx)\n",
                     static_ms, stealing_ms, static_ms / stealing_ms);
        // Not asserted as a hard pass/fail threshold (real wall-clock
        // timing on a shared CI/dev machine is noisy enough that a tight
        // bound would be flaky) -- the number is printed either way, and
        // this is the honest way to report a timing comparison rather
        // than assert a specific speedup that might not reproduce on
        // every machine this runs on.
    }
}

int main() {
    test_range_coder();
    test_lz_matcher();
    test_lz_codec();
    test_bwt_transform();
    test_bwt_sais_matches_reference();
    test_bwt_codec();
    test_pantograph_lift();
    test_pantograph_lift_lossy();
    test_rod_joint_2d();
    test_rod_joint_3d();
    test_rod_joint_3d_similarity();
    test_rod_joint_2d_lossy();
    test_rod_joint_3d_lossy();
    test_quaternion_joint();
    test_quaternion_joint_lossy();
    test_rod_joint_3d_adaptive_blocking();
    test_quaternion_joint_adaptive_blocking();
    test_thread_pool();
    test_work_stealing_pool();
    test_simd_max_abs_diff();
    test_simd_max_abs_diff_cuda();
    test_quat_calibration_cuda_matches_cpu();
    test_pose_codec();
    test_pose_stream();
    test_rans_coder();
    test_codec();
    test_codec_adaptive_skip();
    test_geo_codec();
    test_geo3d_lag_search_toroidal();
    test_rod_joint_lag_search_period_drift();
    test_geo3d_codec_autoselect();

    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
