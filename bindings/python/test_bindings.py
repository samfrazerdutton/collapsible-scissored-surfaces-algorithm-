#!/usr/bin/env python3
"""Smoke test for the Python ctypes bindings (csa.py), run against the
actual built shared library -- not a mock, a real end-to-end check that
the FFI boundary (argument marshaling, struct layout, buffer ownership)
works correctly from Python, not just from C++.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import csa

failures = 0
checks = 0


def check(cond, label):
    global failures, checks
    checks += 1
    if not cond:
        failures += 1
        print(f"FAIL: {label}")


def test_general():
    data = (b"the quick brown fox jumps over the lazy dog. " * 500)
    compressed = csa.compress(data)
    check(len(compressed) > 0 and len(compressed) < len(data) // 10, "general compress ratio")
    back = csa.decompress(compressed)
    check(back == data, "general round-trip")


def test_geo2d():
    points = []
    x, y, dx, dy = 50.0, 0.0, 3.0, 0.0
    for _ in range(300):
        points.append((round(x), round(y)))
        ndx = 1.02 * (dx * math.cos(0.15) - dy * math.sin(0.15))
        ndy = 1.02 * (dx * math.sin(0.15) + dy * math.cos(0.15))
        dx, dy = ndx, ndy
        x += dx
        y += dy

    blob = csa.compress_geo2d(points)
    check(len(blob) > 0, "geo2d compress produced output")
    back = csa.decompress_geo2d(blob)
    check(back == points, "geo2d exact round-trip")


def test_geo2d_lossy():
    points = []
    x, y, heading = 0.0, 0.0, 0.0
    seed = 7
    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return (seed % 10000) / 10000.0
    for _ in range(2000):
        heading += (rnd() - 0.5) * 0.1
        speed = 8.0 + (rnd() - 0.5)
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
        points.append((round(x), round(y)))

    lossless = csa.compress_geo2d(points)
    lossy = csa.compress_geo2d_lossy(points, quant_step=20, resync_interval=64)
    check(len(lossy) < len(lossless), f"lossy smaller than lossless ({len(lossy)} < {len(lossless)})")

    back = csa.decompress_geo2d(lossy)
    check(len(back) == len(points), "lossy round-trip length matches")
    max_err = max(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a, b in zip(points, back))
    check(max_err <= 20 * 64, f"lossy error bounded (max_err={max_err})")
    print(f"  lossy: {len(lossless)} -> {len(lossy)} bytes, max coordinate error = {max_err}")


def test_geo3d_lossy():
    points = []
    x, y, z, heading = 0.0, 0.0, 0.0, 0.0
    seed = 314159
    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return (seed % 10000) / 10000.0
    for _ in range(2000):
        heading += (rnd() - 0.5) * 0.08
        speed = 8.0 + (rnd() - 0.5)
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
        z += 3.0 + (rnd() - 0.5) * 0.5
        points.append((round(x), round(y), round(z)))

    lossless = csa.compress_geo3d(points)
    back = csa.decompress_geo3d(lossless)
    check(back == points, "geo3d exact round-trip")

    lossy = csa.compress_geo3d_lossy(points, quant_step=20, resync_interval=64)
    check(len(lossy) < len(lossless), f"3d lossy smaller than lossless ({len(lossy)} < {len(lossless)})")

    lossy_back = csa.decompress_geo3d(lossy)
    check(len(lossy_back) == len(points), "3d lossy round-trip length matches")
    max_err = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]), abs(a[2] - b[2])) for a, b in zip(points, lossy_back))
    check(max_err <= 20 * 64, f"3d lossy error bounded (max_err={max_err})")
    print(f"  3d lossy: {len(lossless)} -> {len(lossy)} bytes, max coordinate error = {max_err}")


def test_pose_lossy():
    poses = []
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    seed = 4242
    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return (seed % 10000) / 10000.0
    z = 0.0
    qscale = 1 << 20
    for i in range(1500):
        t = i * 0.05
        x, y = 2000 * math.cos(t), 2000 * math.sin(t)
        z += 4.0 + (rnd() - 0.5) * 0.5
        position = (round(x), round(y), round(z))
        orientation = (round(qw * qscale), round(qx * qscale), round(qy * qscale), round(qz * qscale))
        poses.append((position, orientation))

        deg = 2.0 + (rnd() - 0.5) * 0.3
        half = math.radians(deg) / 2.0
        dqw, dqz = math.cos(half), math.sin(half)
        nw = qw * dqw - qz * dqz
        nx = qx * dqw + qy * dqz
        ny = qy * dqw - qx * dqz
        nz = qw * dqz + qz * dqw
        qw, qx, qy, qz = nw, nx, ny, nz

    lossless = csa.compress_pose(poses)
    back = csa.decompress_pose(lossless)
    check(back == poses, "pose exact round-trip")

    lossy = csa.compress_pose_lossy(poses, pos_quant_step=8, pos_resync_interval=64,
                                     quat_quant_step=32, quat_resync_interval=32)
    check(len(lossy) < len(lossless), f"pose lossy smaller than lossless ({len(lossy)} < {len(lossless)})")

    lossy_back = csa.decompress_pose(lossy)
    check(len(lossy_back) == len(poses), "pose lossy round-trip length matches")
    max_err = max(max(abs(a - b) for a, b in zip(pa + oa, pb + ob))
                  for (pa, oa), (pb, ob) in zip(poses, lossy_back))
    check(max_err <= 64 * 64, f"pose lossy error bounded (max_err={max_err})")
    print(f"  pose lossy: {len(lossless)} -> {len(lossy)} bytes, max component error = {max_err}")


def test_error_handling():
    try:
        csa.decompress(b"\x01\x02\x03\x04\x05")
        check(False, "expected CsaError on garbage input")
    except csa.CsaError as e:
        check(len(str(e)) > 0, "CsaError has a message")


def main():
    test_general()
    test_geo2d()
    test_geo2d_lossy()
    test_geo3d_lossy()
    test_pose_lossy()
    test_error_handling()
    print(f"cuda_available() = {csa.cuda_available()}")
    print(f"{checks} checks, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
