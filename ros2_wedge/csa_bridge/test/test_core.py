"""Tests csa_bridge/core.py against the real built libcsa -- no ROS2
required. Run directly: python ros2_wedge/test/test_core.py
(or via pytest, if installed).
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from csa_bridge import PoseBatchCompressor, PoseBatchDecompressor

g_checks = 0
g_failures = 0


def check(cond, msg=""):
    global g_checks, g_failures
    g_checks += 1
    if not cond:
        g_failures += 1
        print(f"FAIL: {msg}")


def make_drone_flight(n=500):
    """Same shape as bench/adversarial_benchmark.py's real datasets: a
    smooth spiral position + a slowly rotating quaternion -- not the real
    data itself, just a reproducible stand-in for a unit test."""
    poses = []
    qw, qz = 1.0, 0.0
    z = 0.0
    for i in range(n):
        t = i * 0.05
        x, y = 2000 * math.cos(t), 2000 * math.sin(t)
        z += 4.0
        poses.append((x, y, z, qw, 0.0, 0.0, qz))
        half = math.radians(2.0) / 2
        dqw, dqz = math.cos(half), math.sin(half)
        qw, qz = qw * dqw - qz * dqz, qw * dqz + qz * dqw
    return poses


def test_empty_flush_returns_none():
    c = PoseBatchCompressor()
    check(c.flush() is None, "flush() on an empty buffer should return None, not an empty blob")


def test_lossless_round_trip():
    poses = make_drone_flight(500)
    c = PoseBatchCompressor(scale=1_000_000, qscale=1_000_000)
    for p in poses:
        c.add_pose(*p)
    check(len(c) == 500, "buffer should hold everything added before flush")
    blob = c.flush()
    check(blob is not None and len(blob) > 0, "flush() should return real compressed bytes")
    check(len(c) == 0, "buffer should be empty immediately after flush")

    d = PoseBatchDecompressor(scale=1_000_000, qscale=1_000_000)
    decoded = d.decode(blob)
    check(len(decoded) == len(poses), "decoded pose count should match what was sent")
    max_err = 0.0
    for (x, y, z, qw, qx, qy, qz), (dx, dy, dz, dqw, dqx, dqy, dqz) in zip(poses, decoded):
        max_err = max(max_err, abs(x - dx), abs(y - dy), abs(z - dz), abs(qw - dqw), abs(qx - dqx), abs(qy - dqy), abs(qz - dqz))
    check(max_err < 1e-5, f"lossless round-trip should match to ~1/scale precision, got max_err={max_err}")

    raw_size = len(poses) * 7 * 8  # float64 per component, the naive baseline
    print(f"  lossless: {len(poses)} poses, naive float64 packing~={raw_size} bytes -> {len(blob)} bytes "
          f"({100*(1-len(blob)/raw_size):.1f}% smaller), max round-trip error={max_err:.2e}")


def test_lossy_round_trip_bounded_error():
    poses = make_drone_flight(500)
    c = PoseBatchCompressor(lossy=True, pos_quant_step=8, pos_resync_interval=64,
                             quat_quant_step=32, quat_resync_interval=64)
    for p in poses:
        c.add_pose(*p)
    blob = c.flush()
    d = PoseBatchDecompressor()
    decoded = d.decode(blob)
    check(len(decoded) == len(poses), "lossy decode should still return every pose")
    max_pos_err = max(abs(a[0] - b[0]) + abs(a[1] - b[1]) + abs(a[2] - b[2]) for a, b in zip(poses, decoded))
    check(max_pos_err < 0.01, f"lossy position error should stay small (bounded by quant_step/scale), got {max_pos_err}")
    print(f"  lossy: {len(blob)} bytes, max summed position error={max_pos_err:.2e}m")


def test_multiple_flushes_are_independent():
    c = PoseBatchCompressor()
    for p in make_drone_flight(10):
        c.add_pose(*p)
    blob1 = c.flush()
    for p in make_drone_flight(20):
        c.add_pose(*p)
    blob2 = c.flush()
    d = PoseBatchDecompressor()
    check(len(d.decode(blob1)) == 10, "first flush should decode back to exactly its own 10 poses")
    check(len(d.decode(blob2)) == 20, "second flush should decode back to exactly its own 20 poses, not 30")


def test_single_pose_batch():
    c = PoseBatchCompressor()
    c.add_pose(1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0)
    blob = c.flush()
    d = PoseBatchDecompressor()
    decoded = d.decode(blob)
    check(len(decoded) == 1, "a batch of exactly one pose should still round-trip")
    check(abs(decoded[0][0] - 1.0) < 1e-6, "single-pose x should round-trip")


if __name__ == "__main__":
    test_empty_flush_returns_none()
    test_lossless_round_trip()
    test_lossy_round_trip_bounded_error()
    test_multiple_flushes_are_independent()
    test_single_pose_batch()
    print(f"{g_checks} checks, {g_failures} failures")
    raise SystemExit(1 if g_failures else 0)
