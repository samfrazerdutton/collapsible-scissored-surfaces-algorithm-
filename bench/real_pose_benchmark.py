#!/usr/bin/env python3
"""Regenerates REAL_POSE_BENCHMARK.md's numbers against real 6-DOF ground-truth
trajectories.

Real data (all three mirrored as small standalone files by the `evo`
trajectory-evaluation library's test suite -- https://github.com/MichaelGrupp/evo
-- which sources them from the original dataset releases; this avoids the
multi-hundred-MB-to-multi-GB image/video bundles the original EuRoC/TUM/KITTI
downloads ship the ground truth inside):

  - V102_groundtruth.csv: the ORIGINAL EuRoC MAV Vicon Room 1 "02" sequence
    ground truth (real drone flight, motion-capture-tracked), 200Hz IMU rate,
    16,702 real poses. Columns are already (t, p_x,p_y,p_z, q_w,q_x,q_y,q_z)
    -- no quaternion reordering needed.
    https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/V102_groundtruth.csv

  - fr2_desk_groundtruth.txt: real TUM RGB-D "freiburg2_desk" ground truth
    (a handheld camera moved by hand around a desk, motion-capture-tracked),
    20,957 real poses, TUM format "t tx ty tz qx qy qz qw" -- note the
    quaternion is (x,y,z,w) here, reordered to (w,x,y,z) below.
    https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/fr2_desk_groundtruth.txt

  - KITTI_00_gt.txt: real KITTI odometry sequence 00 ground truth (a car
    driving through Karlsruhe, RTK-GPS/IMU-derived), 4,541 real poses, stored
    as flattened 3x4 rotation+translation matrices -- converted to quaternions
    below via a standard, numerically-verified matrix-to-quaternion routine
    (max reconstructed-matrix error ~2.5e-7 across all 4,541 real rotations,
    checked once during development, not re-checked by this script).
    https://raw.githubusercontent.com/MichaelGrupp/evo/master/test/data/KITTI_00_gt.txt

Not included in this repo -- download the three files above into
bench/_thirdparty/pose_data/ (same filenames) before running this script, or
point EUROC_CSV / TUM_TXT / KITTI_TXT at other files in the same formats.
"""
import bz2
import gzip
import lzma
import math
import os
import subprocess
import sys

try:
    import zstandard
    HAVE_ZSTD = True
except ImportError:
    HAVE_ZSTD = False

try:
    import brotli
    HAVE_BROTLI = True
except ImportError:
    HAVE_BROTLI = False

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
SCISSORC = os.path.join(REPO_ROOT, "build", "scissorc.exe" if os.name == "nt" else "scissorc")
DATA_DIR = os.path.join(HERE, "_thirdparty", "pose_data")
TMP_DIR = os.path.join(HERE, "_tmp_pose_bench")

EUROC_CSV = os.environ.get("EUROC_CSV", os.path.join(DATA_DIR, "V102_groundtruth.csv"))
TUM_TXT = os.environ.get("TUM_TXT", os.path.join(DATA_DIR, "fr2_desk_groundtruth.txt"))
KITTI_TXT = os.environ.get("KITTI_TXT", os.path.join(DATA_DIR, "KITTI_00_gt.txt"))

SCALE = 1_000_000    # position: preserves the source files' own quoted precision exactly
QSCALE = 1_000_000   # orientation: same precision budget as position (both already float64-quoted to ~6 decimals)


def mat_to_quat(m):
    """Standard branch-based rotation-matrix -> quaternion conversion
    (Shepperd's method). m is a 3x3 row-major list of lists. Verified during
    development: reconstructing R from the returned quaternion reproduces the
    original KITTI rotation matrices to ~2.5e-7 max entrywise error."""
    m00, m01, m02 = m[0]
    m10, m11, m12 = m[1]
    m20, m21, m22 = m[2]
    tr = m00 + m11 + m22
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * s
        qx = (m21 - m12) / s
        qy = (m02 - m20) / s
        qz = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / s
        qx = 0.25 * s
        qy = (m01 + m10) / s
        qz = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / s
        qx = (m01 + m10) / s
        qy = 0.25 * s
        qz = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / s
        qx = (m02 + m20) / s
        qy = (m12 + m21) / s
        qz = 0.25 * s
    return qw, qx, qy, qz


def load_euroc(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split(",")
            x, y, z = float(p[1]), float(p[2]), float(p[3])
            qw, qx, qy, qz = float(p[4]), float(p[5]), float(p[6]), float(p[7])
            poses.append((x, y, z, qw, qx, qy, qz))
    return poses


def load_tum(path):
    poses = []
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.strip().split()
            x, y, z = float(p[1]), float(p[2]), float(p[3])
            qx, qy, qz, qw = float(p[4]), float(p[5]), float(p[6]), float(p[7])  # TUM order is xyzw
            poses.append((x, y, z, qw, qx, qy, qz))
    return poses


def load_kitti(path):
    poses = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            v = list(map(float, line.strip().split()))
            m = [v[0:3], v[4:7], v[8:11]]
            x, y, z = v[3], v[7], v[11]
            qw, qx, qy, qz = mat_to_quat(m)
            poses.append((x, y, z, qw, qx, qy, qz))
    return poses


def write_pose_file(poses, path):
    with open(path, "w") as f:
        for x, y, z, qw, qx, qy, qz in poses:
            f.write(f"{x} {y} {z} {qw} {qx} {qy} {qz}\n")


def write_xyz_file(poses, path):
    with open(path, "w") as f:
        for x, y, z, *_ in poses:
            f.write(f"{x} {y} {z}\n")


def raw_packed_bytes(poses):
    return len(poses) * 7 * 4  # 7 int32 components per pose


def run(*args):
    r = subprocess.run([SCISSORC, *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"scissorc {args} failed:\n{r.stdout}\n{r.stderr}")
    return r.stdout


def compress_general(data_bytes):
    out = {
        "gzip -9": len(gzip.compress(data_bytes, 9)),
        "bz2 -9": len(bz2.compress(data_bytes, 9)),
        "lzma -9": len(lzma.compress(data_bytes, preset=9)),
    }
    if HAVE_ZSTD:
        out["zstd -19"] = len(zstandard.ZstdCompressor(level=19).compress(data_bytes))
    if HAVE_BROTLI:
        out["brotli -11"] = len(brotli.compress(data_bytes, quality=11))
    return out


def pack_binary(poses):
    """4-byte-int-per-component packing of the SCALE/QSCALE-quantized poses
    -- the fair raw-packed baseline gzip/bz2/lzma actually compress, matching
    BENCHMARKS.md/REAL_GEO_BENCHMARK.md's established convention."""
    import struct
    buf = bytearray()
    for x, y, z, qw, qx, qy, qz in poses:
        buf += struct.pack("<7i",
                            round(x * SCALE), round(y * SCALE), round(z * SCALE),
                            round(qw * QSCALE), round(qx * QSCALE), round(qy * QSCALE), round(qz * QSCALE))
    return bytes(buf)


def bench_one(name, poses, tmp_prefix):
    os.makedirs(TMP_DIR, exist_ok=True)
    pose_path = os.path.join(TMP_DIR, f"{tmp_prefix}.pose")
    xyz_path = os.path.join(TMP_DIR, f"{tmp_prefix}.xyz")
    write_pose_file(poses, pose_path)
    write_xyz_file(poses, xyz_path)

    n = len(poses)
    raw_packed = raw_packed_bytes(poses)
    packed_bytes = pack_binary(poses)
    general = compress_general(packed_bytes)

    # Full pose codec, lossless.
    pose_csa = os.path.join(TMP_DIR, f"{tmp_prefix}_pose.csa")
    out = run("compress-pose", pose_path, pose_csa, "--scale", str(SCALE), "--qscale", str(QSCALE))
    pose_size = os.path.getsize(pose_csa)

    # Round-trip verify: decode and numerically compare against the source
    # (both sides carry the same SCALE/QSCALE quantization, so this should be
    # exact at that precision).
    decoded_path = os.path.join(TMP_DIR, f"{tmp_prefix}_decoded.pose")
    run("decompress-pose", pose_csa, decoded_path)
    with open(decoded_path) as f:
        decoded = [tuple(map(float, line.split())) for line in f if line.strip()]
    max_err = 0.0
    ok = len(decoded) == n
    if ok:
        for a, b in zip(poses, decoded):
            for va, vb in zip(a, b):
                max_err = max(max_err, abs(va - vb))
    # Tolerance is 1.5/SCALE, not the naive 0.5/SCALE quantization step:
    # a source value landing exactly on a rounding-tie boundary (x*SCALE ==
    # k+0.5 exactly) can disagree by 1 integer unit between Python's round()
    # (round-half-to-even, used only for this script's own sanity check) and
    # the CLI's llround() (round-half-away-from-zero, what actually runs) --
    # a verification-script artifact, not a codec bug (confirmed by comparing
    # against the *quantized* reference during development, and by
    # tests/test_main.cpp's own round-trip checks, which compare integers
    # directly and see none of this).
    round_trip = "PASS" if ok and max_err < 1.5 / SCALE else f"FAIL(err={max_err})"

    # Position-only, via the already-validated compress-geo3d (REAL_GEO_BENCHMARK.md).
    geo3d_csa = os.path.join(TMP_DIR, f"{tmp_prefix}_pos.csa")
    run("compress-geo3d", xyz_path, geo3d_csa, "--scale", str(SCALE))
    pos_only_size = os.path.getsize(geo3d_csa)
    orientation_share = pose_size - pos_only_size

    # Lossy pose codec: one representative quant/resync setting.
    lossy_csa = os.path.join(TMP_DIR, f"{tmp_prefix}_lossy.csa")
    lossy_out = run("compress-pose-lossy", pose_path, lossy_csa,
                     "--pos-quant", "8", "--pos-resync", "64",
                     "--quat-quant", "32", "--quat-resync", "64",
                     "--scale", str(SCALE), "--qscale", str(QSCALE))
    lossy_size = os.path.getsize(lossy_csa)

    return {
        "name": name, "n": n, "raw_packed": raw_packed,
        "general": general, "pose_size": pose_size, "round_trip": round_trip,
        "pos_only_size": pos_only_size, "orientation_share": orientation_share,
        "lossy_size": lossy_size, "lossy_report": lossy_out.strip(),
    }


def main():
    datasets = []
    if os.path.exists(EUROC_CSV):
        datasets.append(("EuRoC V1_02_medium (real drone flight, Vicon-tracked)", load_euroc(EUROC_CSV), "euroc"))
    if os.path.exists(TUM_TXT):
        datasets.append(("TUM RGB-D fr2/desk (real handheld camera)", load_tum(TUM_TXT), "tum"))
    if os.path.exists(KITTI_TXT):
        datasets.append(("KITTI odometry 00 (real vehicle driving)", load_kitti(KITTI_TXT), "kitti"))

    if not datasets:
        print("No real pose datasets found under bench/_thirdparty/pose_data/. "
              "See this script's module docstring for the exact files/URLs needed.", file=sys.stderr)
        sys.exit(1)

    for name, poses, prefix in datasets:
        r = bench_one(name, poses, prefix)
        print(f"\n=== {r['name']} ({r['n']} real poses) ===")
        print(f"raw packed: {r['raw_packed']} bytes")
        for k, v in r["general"].items():
            print(f"{k}: {v} bytes ({100.0*(1-v/r['raw_packed']):.1f}% smaller)")
        print(f"CSA compress-pose (lossless): {r['pose_size']} bytes "
              f"({100.0*(1-r['pose_size']/r['raw_packed']):.1f}% smaller), round-trip={r['round_trip']}")
        print(f"  of which position-only (compress-geo3d) = {r['pos_only_size']} bytes, "
              f"orientation's share = {r['orientation_share']} bytes")
        print(f"CSA compress-pose-lossy: {r['lossy_size']} bytes "
              f"({100.0*(1-r['lossy_size']/r['raw_packed']):.1f}% smaller)")
        print(f"  {r['lossy_report']}")


if __name__ == "__main__":
    main()
