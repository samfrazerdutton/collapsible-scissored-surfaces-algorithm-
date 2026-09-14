#include "csa/codec.hpp"
#include "csa/bwt_codec.hpp"
#include "csa/lz_codec.hpp"
#include "csa/pantograph_lift.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include "csa/range_coder.hpp"
#include <cstring>

namespace csa {

namespace {

constexpr char kMagic[4] = {'C', 'S', 'A', '1'};

void write_magic_mode(std::vector<u8>& out, Mode mode) {
    out.push_back((u8)kMagic[0]);
    out.push_back((u8)kMagic[1]);
    out.push_back((u8)kMagic[2]);
    out.push_back((u8)kMagic[3]);
    out.push_back((u8)mode);
}

Mode read_magic_mode(const u8* data, size_t size, size_t& pos) {
    if (size < 5 || data[0] != (u8)kMagic[0] || data[1] != (u8)kMagic[1] ||
        data[2] != (u8)kMagic[2] || data[3] != (u8)kMagic[3]) {
        throw std::runtime_error("csa: bad magic / not a CSA1 stream");
    }
    pos = 5;
    return (Mode)data[4];
}

// Serializes a LiftResult: header fields + one range-coded stream holding
// the block ratios, block offsets, and residuals (in that order). Per-
// level block counts are not stored explicitly -- they are a deterministic
// function of original_length (via padded_length and kPantographBlockSize),
// recomputed identically on decode. Ratios/offsets are varint+zigzag coded
// like residuals rather than stored as fixed-width fields: with a small
// per-block calibration there can be thousands of blocks, and letting the
// range coder exploit their (often highly repetitive) small values keeps
// that parameter stream from swamping the actual savings.
void serialize_lift(const LiftResult& lr, std::vector<u8>& out) {
    put_u64(out, lr.original_length);
    put_u32(out, (u32)lr.residuals.size());
    if (lr.original_length > 0) {
        put_i32(out, lr.base.empty() ? 0 : lr.base[0]);
    }

    std::vector<u8> flat_varint;
    for (const auto& level : lr.block_ratios)
        for (i64 ratio : level) write_varint(flat_varint, zigzag_encode64(ratio));
    for (const auto& level : lr.block_offsets)
        for (i32 offset : level) write_varint(flat_varint, zigzag_encode32(offset));
    for (const auto& level : lr.residuals)
        for (i32 v : level) write_varint(flat_varint, zigzag_encode32(v));

    std::vector<u8> coded = range_encode_bytes(flat_varint);
    put_u64(out, (u64)flat_varint.size());
    put_u64(out, (u64)coded.size());
    out.insert(out.end(), coded.begin(), coded.end());
}

LiftResult deserialize_lift(const u8* data, size_t size, size_t& pos) {
    LiftResult lr;
    lr.original_length = get_u64(data, size, pos);
    u32 num_levels = get_u32(data, size, pos);

    // Per-level pair counts (deterministic from original_length), needed
    // up front to know how many blocks of ratio/offset each level wrote.
    size_t padded_length = 1;
    while (padded_length < lr.original_length) padded_length <<= 1;
    std::vector<size_t> level_pair_counts(num_levels);
    {
        size_t level_size = padded_length / 2;
        for (u32 i = 0; i < num_levels; i++) {
            level_pair_counts[i] = level_size;
            level_size /= 2;
        }
    }

    std::vector<size_t> level_block_counts(num_levels);
    for (u32 i = 0; i < num_levels; i++)
        level_block_counts[i] = (level_pair_counts[i] + kPantographBlockSize - 1) / kPantographBlockSize;

    i32 base_value = 0;
    if (lr.original_length > 0) {
        base_value = get_i32(data, size, pos);
        lr.base.push_back(base_value);
    }

    u64 raw_len = get_u64(data, size, pos);
    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated lift payload");
    std::vector<u8> flat_varint = range_decode_bytes(data + pos, (size_t)coded_len, (size_t)raw_len);
    pos += (size_t)coded_len;

    size_t vpos = 0;
    lr.block_ratios.resize(num_levels);
    for (u32 i = 0; i < num_levels; i++) {
        lr.block_ratios[i].resize(level_block_counts[i]);
        for (size_t b = 0; b < level_block_counts[i]; b++) {
            u64 zz = read_varint(flat_varint.data(), flat_varint.size(), vpos);
            lr.block_ratios[i][b] = zigzag_decode64(zz);
        }
    }
    lr.block_offsets.resize(num_levels);
    for (u32 i = 0; i < num_levels; i++) {
        lr.block_offsets[i].resize(level_block_counts[i]);
        for (size_t b = 0; b < level_block_counts[i]; b++) {
            u64 zz = read_varint(flat_varint.data(), flat_varint.size(), vpos);
            lr.block_offsets[i][b] = zigzag_decode32((u32)zz);
        }
    }

    lr.residuals.resize(num_levels);
    for (u32 lvl = 0; lvl < num_levels; lvl++) {
        lr.residuals[lvl].resize(level_pair_counts[lvl]);
        for (size_t i = 0; i < level_pair_counts[lvl]; i++) {
            u64 zz = read_varint(flat_varint.data(), flat_varint.size(), vpos);
            lr.residuals[lvl][i] = zigzag_decode32((u32)zz);
        }
    }
    return lr;
}

// Serializes a RodJoint2DResult: fixed header fields, then a single
// range-coded blob holding the per-block complex ratios followed by the
// residual_x/residual_y streams (same rationale as serialize_lift: block
// counts can be in the thousands, so they ride the entropy coder too
// rather than being stored as fixed-width fields).
void serialize_geo2d(const RodJoint2DResult& r, std::vector<u8>& out) {
    put_u64(out, r.count);
    put_i32(out, r.anchor.x);
    put_i32(out, r.anchor.y);
    put_u32(out, r.quant_step);
    put_u32(out, r.resync_interval);

    std::vector<u8> flat;
    for (u32 v : r.block_lag) write_varint(flat, v);
    for (i64 v : r.block_ratio_re) write_varint(flat, zigzag_encode64(v));
    for (i64 v : r.block_ratio_im) write_varint(flat, zigzag_encode64(v));
    for (i32 v : r.residual_x) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_y) write_varint(flat, zigzag_encode32(v));
    std::vector<u8> coded = range_encode_bytes(flat);
    put_u64(out, (u64)flat.size());
    put_u64(out, (u64)coded.size());
    out.insert(out.end(), coded.begin(), coded.end());
}

RodJoint2DResult deserialize_geo2d(const u8* data, size_t size, size_t& pos) {
    RodJoint2DResult r;
    r.count = get_u64(data, size, pos);
    r.anchor.x = get_i32(data, size, pos);
    r.anchor.y = get_i32(data, size, pos);
    r.quant_step = get_u32(data, size, pos);
    r.resync_interval = get_u32(data, size, pos);

    u64 raw_len = get_u64(data, size, pos);
    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated geo2d payload");
    std::vector<u8> flat = range_decode_bytes(data + pos, (size_t)coded_len, (size_t)raw_len);
    pos += (size_t)coded_len;

    size_t n = (r.count > 0) ? (size_t)(r.count - 1) : 0;
    size_t nblocks = (n + kRodJointBlockSize - 1) / kRodJointBlockSize;
    r.block_lag.resize(nblocks);
    r.block_ratio_re.resize(nblocks);
    r.block_ratio_im.resize(nblocks);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    size_t vpos = 0;
    for (size_t b = 0; b < nblocks; b++)
        r.block_lag[b] = (u32)read_varint(flat.data(), flat.size(), vpos);
    for (size_t b = 0; b < nblocks; b++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.block_ratio_re[b] = zigzag_decode64(zz);
    }
    for (size_t b = 0; b < nblocks; b++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.block_ratio_im[b] = zigzag_decode64(zz);
    }
    for (size_t i = 0; i < n; i++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.residual_x[i] = zigzag_decode32((u32)zz);
    }
    for (size_t i = 0; i < n; i++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.residual_y[i] = zigzag_decode32((u32)zz);
    }
    return r;
}

// Serializes a RodJoint3DSimResult: header fields, then a single range-coded
// blob holding the per-block 3x3 matrices followed by the x/y/z residual
// streams (same rationale as the other serialize_* helpers).
void serialize_geo3d_sim(const RodJoint3DSimResult& r, std::vector<u8>& out) {
    put_u64(out, r.count);
    put_i32(out, r.anchor.x);
    put_i32(out, r.anchor.y);
    put_i32(out, r.anchor.z);
    put_u32(out, r.quant_step);
    put_u32(out, r.resync_interval);

    std::vector<u8> flat;
    for (u32 v : r.block_lag) write_varint(flat, v);
    for (const auto& m : r.block_matrix)
        for (i64 v : m) write_varint(flat, zigzag_encode64(v));
    for (i32 v : r.residual_x) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_y) write_varint(flat, zigzag_encode32(v));
    for (i32 v : r.residual_z) write_varint(flat, zigzag_encode32(v));
    std::vector<u8> coded = range_encode_bytes(flat);
    put_u64(out, (u64)flat.size());
    put_u64(out, (u64)coded.size());
    out.insert(out.end(), coded.begin(), coded.end());
}

RodJoint3DSimResult deserialize_geo3d_sim(const u8* data, size_t size, size_t& pos) {
    RodJoint3DSimResult r;
    r.count = get_u64(data, size, pos);
    r.anchor.x = get_i32(data, size, pos);
    r.anchor.y = get_i32(data, size, pos);
    r.anchor.z = get_i32(data, size, pos);
    r.quant_step = get_u32(data, size, pos);
    r.resync_interval = get_u32(data, size, pos);

    u64 raw_len = get_u64(data, size, pos);
    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated geo3d-sim payload");
    std::vector<u8> flat = range_decode_bytes(data + pos, (size_t)coded_len, (size_t)raw_len);
    pos += (size_t)coded_len;

    size_t n = (r.count > 0) ? (size_t)(r.count - 1) : 0;
    size_t nblocks = (n + kRodJoint3DBlockSize - 1) / kRodJoint3DBlockSize;
    r.block_lag.resize(nblocks);
    r.block_matrix.resize(nblocks);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    r.residual_z.resize(n);
    size_t vpos = 0;
    for (size_t b = 0; b < nblocks; b++)
        r.block_lag[b] = (u32)read_varint(flat.data(), flat.size(), vpos);
    for (size_t b = 0; b < nblocks; b++)
        for (int k = 0; k < 9; k++) {
            u64 zz = read_varint(flat.data(), flat.size(), vpos);
            r.block_matrix[b][k] = zigzag_decode64(zz);
        }
    for (size_t i = 0; i < n; i++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.residual_x[i] = zigzag_decode32((u32)zz);
    }
    for (size_t i = 0; i < n; i++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.residual_y[i] = zigzag_decode32((u32)zz);
    }
    for (size_t i = 0; i < n; i++) {
        u64 zz = read_varint(flat.data(), flat.size(), vpos);
        r.residual_z[i] = zigzag_decode32((u32)zz);
    }
    return r;
}

} // namespace

namespace {
// Below this size, trying every candidate always costs so little
// (milliseconds) that there's no reason to skip any of them "to save
// time" -- the adaptive skip below only kicks in where it can actually
// matter.
constexpr size_t kAdaptiveSizeThreshold = 100'000;
// If the LZ candidate already compressed to less than this fraction of
// the input, that's strong *measured* evidence (not a guessed content
// type) that the data has exploitable repeated-substring structure --
// Pantograph Lift's predictive model is very unlikely to beat a result
// already this strong, so it's skipped to save real time on large
// inputs. This is a real, if imperfect, tradeoff: it is conceivable for
// an adversarial file to have LZ do reasonably well *and* have Pantograph
// Lift do even better, in which case this heuristic gives up a small,
// unmeasured amount of ratio for a real, measured amount of speed. It
// never runs below kAdaptiveSizeThreshold, and it is always based on
// this file's actual LZ result, never a guess from a file extension or a
// sniffed "looks like text" classifier.
constexpr double kAdaptiveLzStrongRatio = 0.35;

// BWT's suffix-array construction cost scales with input size regardless
// of whether it ends up winning (unlike the LZ-ratio-based skip above,
// which is a real signal specifically for Pantograph Lift) -- measured on
// an 18MB real-source-code corpus, trying it unconditionally cost an
// extra ~10s at the "fast" level alone without winning at any size, while
// on realistic few-hundred-KB-to-1MB files (server logs, JSON telemetry,
// sensor CSV -- see USE_CASES.md) it won convincingly (8-24% smaller than
// the next-best candidate) for negligible added time. This is a plain
// size cutoff, not a content-based guess: below it, BWT is always tried
// (it either wins or it doesn't, cheaply); above it, the cost of trying
// stops being worth paying for files this size actually measured.
constexpr size_t kBwtMaxInputSize = 4 * 1024 * 1024;
} // namespace

std::vector<u8> compress(const std::vector<u8>& input, bool use_gpu, int lz_max_chain, size_t lz_nice_length) {
    // RAW candidate.
    std::vector<u8> raw_blob;
    write_magic_mode(raw_blob, Mode::Raw);
    put_u64(raw_blob, (u64)input.size());
    raw_blob.insert(raw_blob.end(), input.begin(), input.end());

    if (input.empty()) return raw_blob;

    // GENERAL-LZ candidate (computed first: its result is also the signal
    // the adaptive heuristic below uses to decide whether Pantograph Lift
    // is worth trying at all).
    std::vector<u8> lz_blob;
    write_magic_mode(lz_blob, Mode::GeneralLZ);
    std::vector<u8> lz_payload = lz_encode(input, lz_max_chain, lz_nice_length);
    lz_blob.insert(lz_blob.end(), lz_payload.begin(), lz_payload.end());

    bool skip_pantograph = false;
    if (input.size() >= kAdaptiveSizeThreshold) {
        double lz_ratio = (double)lz_blob.size() / (double)input.size();
        skip_pantograph = lz_ratio < kAdaptiveLzStrongRatio;
    }

    // GENERAL-BWT candidate: a different kind of redundancy than the LZ
    // matcher's exact-repeat matching -- local byte-context statistics,
    // the kind bz2's Burrows-Wheeler stage targets. Skipped above
    // kBwtMaxInputSize on a plain size cutoff (see its comment) rather
    // than the LZ-ratio signal used for Pantograph Lift below: BWT's
    // strength doesn't correlate with how well LZ already did (it beat an
    // already-strong LZ result by 24% on one real use case), so that
    // signal isn't a valid skip predictor for it.
    const std::vector<u8>* best = &raw_blob;
    if (lz_blob.size() < best->size()) best = &lz_blob;

    std::vector<u8> bwt_blob;
    if (input.size() <= kBwtMaxInputSize) {
        write_magic_mode(bwt_blob, Mode::GeneralBWT);
        std::vector<u8> bwt_payload = bwt_encode(input);
        bwt_blob.insert(bwt_blob.end(), bwt_payload.begin(), bwt_payload.end());
        if (bwt_blob.size() < best->size()) best = &bwt_blob;
    }

    std::vector<u8> general_blob;
    if (!skip_pantograph) {
        std::vector<i32> as_i32(input.size());
        for (size_t i = 0; i < input.size(); i++) as_i32[i] = (i32)input[i];
        LiftResult lr;
        bool used_gpu = false;
        if (use_gpu) {
            // thread_local: each thread gets its own persistent GPU
            // buffers (CudaLiftSession isn't safe to share across
            // threads), but repeated compress() calls on the *same*
            // thread -- the common case for a long-lived worker -- reuse
            // them instead of paying cudaMalloc/cudaFree on every call.
            static thread_local CudaLiftSession session;
            used_gpu = session.forward(as_i32, lr);
        }
        if (!used_gpu) lr = pantograph_lift_forward(as_i32);

        write_magic_mode(general_blob, Mode::General);
        serialize_lift(lr, general_blob);
        if (general_blob.size() < best->size()) best = &general_blob;
    }
    return *best;
}

std::vector<u8> decompress(const std::vector<u8>& blob) {
    size_t pos = 0;
    Mode mode = read_magic_mode(blob.data(), blob.size(), pos);
    if (mode == Mode::Raw) {
        u64 len = get_u64(blob.data(), blob.size(), pos);
        if (pos + len > blob.size()) throw std::runtime_error("csa: truncated raw payload");
        return std::vector<u8>(blob.begin() + pos, blob.begin() + pos + len);
    }
    if (mode == Mode::General) {
        LiftResult lr = deserialize_lift(blob.data(), blob.size(), pos);
        std::vector<i32> vals = pantograph_lift_inverse(lr);
        std::vector<u8> out(vals.size());
        for (size_t i = 0; i < vals.size(); i++) out[i] = (u8)vals[i];
        return out;
    }
    if (mode == Mode::GeneralLZ) {
        return lz_decode(blob.data(), blob.size(), pos);
    }
    if (mode == Mode::GeneralBWT) {
        return bwt_decode(blob.data(), blob.size(), pos);
    }
    throw std::runtime_error("csa: decompress() called on a geometric-mode stream; use decompress_geo2d/3d");
}

// Tries both lag strategies and keeps whichever actually serializes
// smaller: per-block auto search (rod_joint_2d_forward's default,
// force_lag == 0 -- picks each calibration block's lag independently by
// least-squares residual SSE) and the best single whole-file-uniform
// forced lag (the older design: one full encode per kRodJointCandidateLags
// candidate, compared by real serialized bytes). Per-block search is a
// strictly finer-grained version of "one lag for the whole file", so it
// wins whenever a path's oscillation period itself changes partway
// through (see test_rod_joint_lag_search_period_drift) or is short enough
// to vary within a single calibration block -- but its selection criterion
// (raw residual SSE, cheap to compute per-candidate-per-block) is a proxy
// for the real objective (entropy-coded size after quantization), and a
// proxy can occasionally diverge from the real thing, especially once
// quantization is involved. Comparing actual serialized bytes against the
// old whole-file search here means this change can only ever match or
// beat the pre-per-block-search behavior, never quietly regress it.
std::vector<u8> best_geo2d_encoding(const std::vector<Point2i>& points, u32 quant_step, u32 resync_interval) {
    RodJoint2DResult auto_r = rod_joint_2d_forward(points, /*force_lag=*/0, quant_step, resync_interval);
    std::vector<u8> best;
    write_magic_mode(best, Mode::Geo2D);
    serialize_geo2d(auto_r, best);

    for (u32 lag : kRodJointCandidateLags) {
        RodJoint2DResult r = rod_joint_2d_forward(points, lag, quant_step, resync_interval);
        std::vector<u8> out;
        write_magic_mode(out, Mode::Geo2D);
        serialize_geo2d(r, out);
        if (out.size() < best.size()) best = std::move(out);
    }
    return best;
}

std::vector<u8> compress_geo2d(const std::vector<Point2i>& points) {
    return best_geo2d_encoding(points, 1, 0);
}

std::vector<Point2i> decompress_geo2d(const std::vector<u8>& blob) {
    size_t pos = 0;
    Mode mode = read_magic_mode(blob.data(), blob.size(), pos);
    if (mode != Mode::Geo2D) throw std::runtime_error("csa: not a Geo2D stream");
    RodJoint2DResult r = deserialize_geo2d(blob.data(), blob.size(), pos);
    return rod_joint_2d_inverse(r); // handles lossy blobs transparently: quant_step/resync_interval ride in the blob itself
}

std::vector<u8> compress_geo2d_lossy(const std::vector<Point2i>& points, u32 quant_step, u32 resync_interval) {
    if (quant_step <= 1) return compress_geo2d(points); // q<=1 has no lossy effect; use the lag-searching lossless path
    return best_geo2d_encoding(points, quant_step, resync_interval);
}

// Geo3D tries two candidate models -- the xy-rotation+z-affine composition
// (best for helix-like constant-radius paths) and the true 3D similarity
// joint (best for paths that genuinely tumble in 3D) -- and keeps
// whichever encodes smaller, tagged with one sub-mode byte. Both are
// lossless regardless of which wins.
enum class Geo3DSubMode : u8 { Composition = 0, Similarity3D = 1 };

// Same auto-vs-best-forced-uniform-lag comparison as best_geo2d_encoding,
// for the xy+z composition model.
std::vector<u8> best_geo3d_composition_encoding(const std::vector<Point3i>& points) {
    RodJoint3DResult auto_comp = rod_joint_3d_forward(points);
    std::vector<u8> best;
    write_magic_mode(best, Mode::Geo3D);
    best.push_back((u8)Geo3DSubMode::Composition);
    serialize_geo2d(auto_comp.xy, best);
    serialize_lift(auto_comp.lift_z, best);

    for (u32 lag : kRodJointCandidateLags) {
        RodJoint3DResult comp = rod_joint_3d_forward(points, lag);
        std::vector<u8> out;
        write_magic_mode(out, Mode::Geo3D);
        out.push_back((u8)Geo3DSubMode::Composition);
        serialize_geo2d(comp.xy, out);
        serialize_lift(comp.lift_z, out);
        if (out.size() < best.size()) best = std::move(out);
    }
    return best;
}

// Same auto-vs-best-forced-uniform-lag comparison, for the true 3D
// similarity joint.
std::vector<u8> best_geo3d_similarity_encoding(const std::vector<Point3i>& points, u32 quant_step, u32 resync_interval) {
    RodJoint3DSimResult auto_sim = rod_joint_3d_similarity_forward(points, /*force_lag=*/0, quant_step, resync_interval);
    std::vector<u8> best;
    write_magic_mode(best, Mode::Geo3D);
    best.push_back((u8)Geo3DSubMode::Similarity3D);
    serialize_geo3d_sim(auto_sim, best);

    for (u32 lag : kRodJointCandidateLags) {
        RodJoint3DSimResult sim = rod_joint_3d_similarity_forward(points, lag, quant_step, resync_interval);
        std::vector<u8> out;
        write_magic_mode(out, Mode::Geo3D);
        out.push_back((u8)Geo3DSubMode::Similarity3D);
        serialize_geo3d_sim(sim, out);
        if (out.size() < best.size()) best = std::move(out);
    }
    return best;
}

std::vector<u8> compress_geo3d(const std::vector<Point3i>& points) {
    std::vector<u8> comp_out = best_geo3d_composition_encoding(points);
    std::vector<u8> sim_out = best_geo3d_similarity_encoding(points, 1, 0);
    return (sim_out.size() < comp_out.size()) ? sim_out : comp_out;
}

// Lossy geo3d only tries the true 3D similarity joint, not the xy+z
// composition -- the composition's z-axis goes through the Pantograph
// Lift, which doesn't have a lossy mode yet (see DESIGN.md's future
// work), so it can't participate in a fair lossy comparison here.
std::vector<u8> compress_geo3d_lossy(const std::vector<Point3i>& points, u32 quant_step, u32 resync_interval) {
    if (quant_step <= 1) return compress_geo3d(points); // no lossy effect; use the full lossless auto-select
    return best_geo3d_similarity_encoding(points, quant_step, resync_interval);
}

std::vector<Point3i> decompress_geo3d(const std::vector<u8>& blob) {
    size_t pos = 0;
    Mode mode = read_magic_mode(blob.data(), blob.size(), pos);
    if (mode != Mode::Geo3D) throw std::runtime_error("csa: not a Geo3D stream");
    if (pos >= blob.size()) throw std::runtime_error("csa: truncated geo3d sub-mode");
    Geo3DSubMode sub = (Geo3DSubMode)blob[pos++];

    if (sub == Geo3DSubMode::Similarity3D) {
        RodJoint3DSimResult r = deserialize_geo3d_sim(blob.data(), blob.size(), pos);
        return rod_joint_3d_similarity_inverse(r);
    }
    RodJoint3DResult r;
    r.xy = deserialize_geo2d(blob.data(), blob.size(), pos);
    r.lift_z = deserialize_lift(blob.data(), blob.size(), pos);
    return rod_joint_3d_inverse(r);
}

} // namespace csa
