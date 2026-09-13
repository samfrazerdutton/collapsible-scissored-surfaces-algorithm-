#include "csa/codec.hpp"
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

    std::vector<u8> flat;
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

    u64 raw_len = get_u64(data, size, pos);
    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated geo2d payload");
    std::vector<u8> flat = range_decode_bytes(data + pos, (size_t)coded_len, (size_t)raw_len);
    pos += (size_t)coded_len;

    size_t n = (r.count > 0) ? (size_t)(r.count - 1) : 0;
    size_t nblocks = (n + kRodJointBlockSize - 1) / kRodJointBlockSize;
    r.block_ratio_re.resize(nblocks);
    r.block_ratio_im.resize(nblocks);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    size_t vpos = 0;
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

    std::vector<u8> flat;
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

    u64 raw_len = get_u64(data, size, pos);
    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated geo3d-sim payload");
    std::vector<u8> flat = range_decode_bytes(data + pos, (size_t)coded_len, (size_t)raw_len);
    pos += (size_t)coded_len;

    size_t n = (r.count > 0) ? (size_t)(r.count - 1) : 0;
    size_t nblocks = (n + kRodJoint3DBlockSize - 1) / kRodJoint3DBlockSize;
    r.block_matrix.resize(nblocks);
    r.residual_x.resize(n);
    r.residual_y.resize(n);
    r.residual_z.resize(n);
    size_t vpos = 0;
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

std::vector<u8> compress(const std::vector<u8>& input, bool use_gpu) {
    // RAW candidate.
    std::vector<u8> raw_blob;
    write_magic_mode(raw_blob, Mode::Raw);
    put_u64(raw_blob, (u64)input.size());
    raw_blob.insert(raw_blob.end(), input.begin(), input.end());

    if (input.empty()) return raw_blob;

    // GENERAL candidate.
    std::vector<i32> as_i32(input.size());
    for (size_t i = 0; i < input.size(); i++) as_i32[i] = (i32)input[i];
    LiftResult lr;
    bool used_gpu = use_gpu && pantograph_lift_forward_cuda(as_i32, lr);
    if (!used_gpu) lr = pantograph_lift_forward(as_i32);

    std::vector<u8> general_blob;
    write_magic_mode(general_blob, Mode::General);
    serialize_lift(lr, general_blob);

    return (general_blob.size() < raw_blob.size()) ? general_blob : raw_blob;
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
    throw std::runtime_error("csa: decompress() called on a geometric-mode stream; use decompress_geo2d/3d");
}

std::vector<u8> compress_geo2d(const std::vector<Point2i>& points) {
    RodJoint2DResult r = rod_joint_2d_forward(points);
    std::vector<u8> out;
    write_magic_mode(out, Mode::Geo2D);
    serialize_geo2d(r, out);
    return out;
}

std::vector<Point2i> decompress_geo2d(const std::vector<u8>& blob) {
    size_t pos = 0;
    Mode mode = read_magic_mode(blob.data(), blob.size(), pos);
    if (mode != Mode::Geo2D) throw std::runtime_error("csa: not a Geo2D stream");
    RodJoint2DResult r = deserialize_geo2d(blob.data(), blob.size(), pos);
    return rod_joint_2d_inverse(r);
}

// Geo3D tries two candidate models -- the xy-rotation+z-affine composition
// (best for helix-like constant-radius paths) and the true 3D similarity
// joint (best for paths that genuinely tumble in 3D) -- and keeps
// whichever encodes smaller, tagged with one sub-mode byte. Both are
// lossless regardless of which wins.
enum class Geo3DSubMode : u8 { Composition = 0, Similarity3D = 1 };

std::vector<u8> compress_geo3d(const std::vector<Point3i>& points) {
    RodJoint3DResult comp = rod_joint_3d_forward(points);
    std::vector<u8> comp_out;
    write_magic_mode(comp_out, Mode::Geo3D);
    comp_out.push_back((u8)Geo3DSubMode::Composition);
    serialize_geo2d(comp.xy, comp_out);
    serialize_lift(comp.lift_z, comp_out);

    RodJoint3DSimResult sim = rod_joint_3d_similarity_forward(points);
    std::vector<u8> sim_out;
    write_magic_mode(sim_out, Mode::Geo3D);
    sim_out.push_back((u8)Geo3DSubMode::Similarity3D);
    serialize_geo3d_sim(sim, sim_out);

    return (sim_out.size() < comp_out.size()) ? sim_out : comp_out;
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
