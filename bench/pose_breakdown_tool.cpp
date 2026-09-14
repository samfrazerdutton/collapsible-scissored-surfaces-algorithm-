// Diagnostic tool (not part of the shipped codec): breaks down where
// compress_pose's bytes actually go -- calibration-parameter overhead vs.
// residual -- on a real .pose file, to inform whether a hierarchical/
// multi-scale Rod-Joint/Quaternion-Joint calibration (mirroring the
// Pantograph Lift's recursive halving structure) would actually pay off
// before spending effort building it. See the "next step" discussion this
// was built to answer, not a permanent part of the library.
//
// Params and residuals are entropy-coded SEPARATELY here (each through its
// own range_encode_bytes call) to get a real per-stream byte count; the
// production codec (codec.cpp's serialize_geo3d_sim/serialize_quat_joint)
// encodes them together in one adaptive pass, so the true production
// split is approximately but not exactly this -- close enough to answer
// "is parameter overhead worth attacking," not a claim about the exact
// production byte count.
//
// Not wired into CMakeLists.txt (deliberately -- this is a one-off
// diagnostic, not a shipped tool). Build manually against the already-
// built csa_core static library, e.g. from an MSVC dev prompt:
//   cl /nologo /EHsc /O2 /MD /std:c++17 /I include
//      bench/pose_breakdown_tool.cpp /Fe:build/pose_breakdown_tool.exe
//      /link build/csa_core.lib
// then run it with name=path.pose arguments, e.g.:
//   build/pose_breakdown_tool.exe "EuRoC"=bench/_tmp_pose_bench/euroc.pose
#include "csa/common.hpp"
#include "csa/quaternion_joint.hpp"
#include "csa/range_coder.hpp"
#include "csa/rod_joint_transform.hpp"
#include <cmath>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace csa;

namespace {

struct Poses {
    std::vector<Point3i> pos;
    std::vector<Quat4i> quat;
};

Poses read_pose_file(const std::string& path, double scale, double qscale) {
    Poses out;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        double x, y, z, qw, qx, qy, qz;
        if (!(ss >> x >> y >> z >> qw >> qx >> qy >> qz)) continue;
        out.pos.push_back({(i32)llround(x * scale), (i32)llround(y * scale), (i32)llround(z * scale)});
        out.quat.push_back({(i32)llround(qw * qscale), (i32)llround(qx * qscale),
                             (i32)llround(qy * qscale), (i32)llround(qz * qscale)});
    }
    return out;
}

size_t coded_size(const std::vector<u8>& flat) {
    return range_encode_bytes(flat).size();
}

// Real per-stream residual magnitude, summarized per calibration block, to
// look for multi-scale structure: if block-to-block residual energy is
// highly variable (a coefficient of variation well above what a single
// smooth trend would produce), a fixed block size is fighting variation
// at more than one timescale -- evidence a coarse-to-fine hierarchy could
// help. If it's fairly flat, the current single scale already matches the
// data and hierarchy would mostly just be overhead.
double block_energy_cv(const std::vector<i32>& res, const std::vector<i32>& res2,
                        const std::vector<i32>& res3, size_t block_size) {
    size_t n = res.size();
    if (n == 0) return 0.0;
    size_t nblocks = (n + block_size - 1) / block_size;
    std::vector<double> block_rms(nblocks, 0.0);
    for (size_t b = 0; b < nblocks; b++) {
        size_t start = b * block_size, end = std::min(n, start + block_size);
        double sse = 0.0;
        for (size_t i = start; i < end; i++) {
            sse += (double)res[i] * res[i] + (double)res2[i] * res2[i] + (double)res3[i] * res3[i];
        }
        block_rms[b] = std::sqrt(sse / (double)(3 * (end - start)));
    }
    double mean = std::accumulate(block_rms.begin(), block_rms.end(), 0.0) / (double)nblocks;
    double var = 0.0;
    for (double v : block_rms) var += (v - mean) * (v - mean);
    var /= (double)nblocks;
    return mean > 1e-9 ? std::sqrt(var) / mean : 0.0;
}

void analyze(const std::string& name, const Poses& poses) {
    std::printf("=== %s (%zu poses) ===\n", name.c_str(), poses.pos.size());

    // Position: the true 3D similarity joint (public API), default
    // per-block auto lag search, lossless.
    RodJoint3DSimResult pr = rod_joint_3d_similarity_forward(poses.pos);
    std::vector<u8> pos_params, pos_residuals;
    for (u32 v : pr.block_lag) write_varint(pos_params, v);
    for (const auto& m : pr.block_matrix)
        for (i64 v : m) write_varint(pos_params, zigzag_encode64(v));
    for (i32 v : pr.residual_x) write_varint(pos_residuals, zigzag_encode32(v));
    for (i32 v : pr.residual_y) write_varint(pos_residuals, zigzag_encode32(v));
    for (i32 v : pr.residual_z) write_varint(pos_residuals, zigzag_encode32(v));
    size_t pos_param_bytes = coded_size(pos_params);
    size_t pos_res_bytes = coded_size(pos_residuals);
    double pos_cv = block_energy_cv(pr.residual_x, pr.residual_y, pr.residual_z, kRodJoint3DBlockSize);

    // Orientation: Quaternion Joint (public API), default per-block auto
    // lag search, lossless.
    QuaternionJointResult qr = quaternion_joint_forward(poses.quat);
    std::vector<u8> quat_params, quat_residuals;
    for (u32 v : qr.block_lag) write_varint(quat_params, v);
    for (const auto& d : qr.block_delta)
        for (i64 v : d) write_varint(quat_params, zigzag_encode64(v));
    for (i32 v : qr.residual_w) write_varint(quat_residuals, zigzag_encode32(v));
    for (i32 v : qr.residual_x) write_varint(quat_residuals, zigzag_encode32(v));
    for (i32 v : qr.residual_y) write_varint(quat_residuals, zigzag_encode32(v));
    for (i32 v : qr.residual_z) write_varint(quat_residuals, zigzag_encode32(v));
    size_t quat_param_bytes = coded_size(quat_params);
    size_t quat_res_bytes = coded_size(quat_residuals);
    double quat_cv = block_energy_cv(qr.residual_w, qr.residual_x, qr.residual_y, kQuatJointBlockSize);
    // (residual_z folded in separately below since block_energy_cv takes 3 streams)
    double quat_cv_z_included = 0.0;
    {
        // recompute including z properly: reuse function with w,x,y then note z separately is awkward;
        // instead just report cv using x,y,z (drop w) as a second estimate for sanity.
        quat_cv_z_included = block_energy_cv(qr.residual_x, qr.residual_y, qr.residual_z, kQuatJointBlockSize);
    }

    size_t pos_blocks = pr.block_lag.size(), quat_blocks = qr.block_lag.size();

    std::printf("  position: %zu blocks, params=%zu bytes, residuals=%zu bytes (params = %.2f%% of position total), block-energy CV=%.2f\n",
                pos_blocks, pos_param_bytes, pos_res_bytes,
                100.0 * (double)pos_param_bytes / (double)(pos_param_bytes + pos_res_bytes), pos_cv);
    std::printf("  orientation: %zu blocks, params=%zu bytes, residuals=%zu bytes (params = %.2f%% of orientation total), block-energy CV(w,x,y)=%.2f CV(x,y,z)=%.2f\n",
                quat_blocks, quat_param_bytes, quat_res_bytes,
                100.0 * (double)quat_param_bytes / (double)(quat_param_bytes + quat_res_bytes), quat_cv, quat_cv_z_included);
    std::printf("\n");
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: pose_breakdown_tool <name1>=<path1.pose> [<name2>=<path2.pose> ...]\n");
        return 1;
    }
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        size_t eq = arg.find('=');
        std::string name = (eq == std::string::npos) ? arg : arg.substr(0, eq);
        std::string path = (eq == std::string::npos) ? arg : arg.substr(eq + 1);
        Poses poses = read_pose_file(path, 1e6, 1e6);
        analyze(name, poses);
    }
    return 0;
}
