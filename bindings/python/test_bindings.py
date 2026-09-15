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


def test_inspect_and_verify():
    # Layer 2 (CSAG) case: compress_geo3d doesn't add Layer 2 framing itself
    # (that's a CLI/browser-only convention -- see FORMAT.md) so build a
    # real Layer-2-wrapped blob the same way cli/main.cpp's squeeze does.
    import struct as _struct
    points = [(1000, 2000, 3000), (1100, 2100, 3100), (1200, 2200, 3200)]
    layer1 = csa.compress_geo3d(points)
    scale = 1000
    wrapped = b"CSAG" + bytes([3]) + _struct.pack("<Q", scale) + layer1

    info = csa.inspect(wrapped)
    check(info["layer"] == 2, "inspect: geo3d wrapped blob is layer 2")
    check(info["dims"] == 3 and info["shape"] == "geo3d", "inspect: dims/shape match")
    check(info["scale"] == scale, "inspect: scale round-trips")
    check("qscale" not in info, "inspect: qscale absent for dims=3")
    check(info["mode_name"] == "Geo3D", "inspect: mode_name matches Layer 1 mode byte")

    result = csa.verify(wrapped)
    check(result["shape"] == "geo3d" and result["count"] == len(points), f"verify: geo3d decodes correctly ({result})")

    # Bare Layer 1 (general-mode) blob: no CSAG wrapper at all.
    general_blob = csa.compress(b"some arbitrary bytes, not a numeric table" * 20)
    info_general = csa.inspect(general_blob)
    check(info_general["layer"] == 1, "inspect: general blob is bare layer 1")
    check(info_general["mode_name"] in ("General", "GeneralLZ", "GeneralBWT", "Raw"), "inspect: general mode_name is a real Mode value")
    result_general = csa.verify(general_blob)
    check(result_general["shape"] == "general", "verify: general blob reports shape=general")

    # Malformed input: neither magic present.
    try:
        csa.inspect(b"not a csa file at all")
        check(False, "expected CsaError for data with no recognized magic")
    except csa.CsaError:
        pass

    # Truncated CSAG header (magic + dims byte but no scale field).
    try:
        csa.inspect(b"CSAG" + bytes([3]) + b"\x00\x00\x00")
        check(False, "expected CsaError for truncated CSAG header")
    except csa.CsaError:
        pass


def test_optimize():
    points3d = [(int(3000 * math.cos(i * 0.11) + (i * 37 % 23) - 11),
                 int(3000 * math.sin(i * 0.11) + (i * 53 % 19) - 9),
                 int(i * 4 + (i * 29 % 17))) for i in range(600)]
    lossless_blob = csa.compress_geo3d(points3d)

    r = csa.optimize_geo3d(points3d, max_pos_error=20)
    check(not r["lossless"], "optimize_geo3d: budget of 20 units is loose enough to find a lossy config")
    check(r["measured_error"] <= 20, f"optimize_geo3d: measured error respects budget ({r['measured_error']})")
    check(len(r["blob"]) < len(lossless_blob), f"optimize_geo3d: found smaller-than-lossless ({len(r['blob'])} < {len(lossless_blob)})")
    back = csa.decompress_geo3d(r["blob"])
    max_err = max(max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(points3d, back))
    check(max_err == r["measured_error"], "optimize_geo3d: reported error matches independently re-measured error")

    poses = []
    qw, qz = 1000000, 0
    for i in range(300):
        poses.append(((1000 + i * 3, 2000 - i * 2, 3000 + i), (qw, 0, 0, qz)))
        qz += 500  # drift the orientation component a little each step

    rp = csa.optimize_pose(poses, max_pos_error=8, max_quat_error=2000)
    check(rp["measured_pos_error"] <= 8, f"optimize_pose: position error respects budget ({rp['measured_pos_error']})")
    check(rp["measured_quat_error"] <= 2000, f"optimize_pose: rotation error respects budget ({rp['measured_quat_error']})")
    back_p = csa.decompress_pose(rp["blob"])
    check(len(back_p) == len(poses), "optimize_pose: round trip preserves record count")

    # Position-only budget: rotation should fall back to quant_step=1 (near-lossless), not be dropped.
    rp2 = csa.optimize_pose(poses, max_pos_error=8)
    check(rp2["quat_quant_step"] in (1, None), "optimize_pose: rotation held at finest step when no rotation budget given")

    try:
        csa.optimize_pose(poses)
        check(False, "expected ValueError with no budget given")
    except ValueError:
        pass


def test_profile():
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for i in range(200):
            f.write(f"{1.0 + i * 0.1:.6f} {2.0 - i * 0.1:.6f} {3.0 + i * 0.2:.6f}\n")
        path = f.name
    try:
        info = csa.profile(path)
        check(info["shape"] == "geo3d", f"profile: detects geo3d for a clean 3-column table ({info})")
        check(info["confidence_pct"] == 100.0, "profile: 100% confidence for a fully clean table")
        check(info["recommended"] is True, "profile: recommends CSA for detected geo3d data")
    finally:
        os.unlink(path)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("this is not spatial data, just prose text with no numeric table structure at all\n")
        path2 = f.name
    try:
        info2 = csa.profile(path2)
        check(info2["shape"] == "general", "profile: detects general shape for non-numeric text")
        check(info2["recommended"] is False, "profile: does not recommend CSA for general text")
    finally:
        os.unlink(path2)


def main():
    test_general()
    test_geo2d()
    test_geo2d_lossy()
    test_geo3d_lossy()
    test_pose_lossy()
    test_error_handling()
    test_inspect_and_verify()
    test_optimize()
    test_profile()
    print(f"cuda_available() = {csa.cuda_available()}")
    print(f"{checks} checks, {failures} failures")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
