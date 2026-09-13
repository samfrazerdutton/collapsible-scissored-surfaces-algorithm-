#!/usr/bin/env python3
"""Generates the test datasets used by run_benchmarks.py.

Two families of data, matching the two transform modes:

  - General-mode (byte-stream) datasets: text, an 8-bit grayscale-style
    smooth gradient (a realistic byte-oriented smooth signal, e.g. one
    scanline of an image), and incompressible random bytes as the
    adversarial control case.

  - Geo-mode (point-sequence) datasets: a perfect logarithmic spiral and a
    toroidal helix (the exact shape classes the Harvard/Tokyo pantograph
    lattice paper calls out as deployable), a noisy GPS-track-like random
    walk, and a synthetic single-ring LiDAR scan (points at roughly
    constant radius stepped around 360 degrees with small elevation
    noise, mimicking one rotation of a spinning automotive LiDAR) as a
    domain-realistic case, plus a pure random walk as the adversarial
    control.
"""
import math
import os
import random
import struct

OUT = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(OUT, exist_ok=True)


def write_bytes(name, data: bytes):
    path = os.path.join(OUT, name)
    with open(path, "wb") as f:
        f.write(data)
    print(f"{name}: {len(data)} bytes")


def write_points(name, points, dims):
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        for p in points:
            f.write(" ".join(f"{v:.6f}" for v in p) + "\n")
    print(f"{name}: {len(points)} points ({dims}D)")


# ---- General-mode datasets ----

def gen_text():
    random.seed(1)
    sentence = "the quick brown fox jumps over the lazy dog while pantograph lattices deploy. "
    text = (sentence * 4000).encode("ascii")
    write_bytes("text_repetitive.bin", text)


def gen_smooth_gradient():
    # One 4096-wide "scanline": smooth low-frequency brightness signal
    # quantized to a byte, like a real image row.
    n = 4096
    data = bytearray(n)
    for i in range(n):
        v = 128 + 100 * math.sin(i * 0.01) + 20 * math.sin(i * 0.2)
        data[i] = max(0, min(255, int(round(v))))
    write_bytes("smooth_gradient.bin", bytes(data))


def gen_random():
    random.seed(2)
    data = bytes(random.randrange(256) for _ in range(65536))
    write_bytes("random.bin", data)


# ---- Geo-mode datasets ----

def gen_spiral_2d():
    # Logarithmic spiral: constant rotation+scale per step by construction.
    # Growth is compounded 1500 times, so scale_step must stay tiny or the
    # radius blows past int32 range once the CLI quantizes by 1000x.
    points = []
    x, y = 100.0, 0.0
    dx, dy = 4.0, 0.0
    angle_step, scale_step = 0.15, 1.003
    for _ in range(1500):
        points.append((x, y))
        ndx = scale_step * (dx * math.cos(angle_step) - dy * math.sin(angle_step))
        ndy = scale_step * (dx * math.sin(angle_step) + dy * math.cos(angle_step))
        dx, dy = ndx, ndy
        x += dx
        y += dy
    write_points("spiral.xy", points, 2)


def gen_helix_3d():
    points = []
    for i in range(6000):
        t = i * 0.08
        x = 500 * math.cos(t)
        y = 500 * math.sin(t)
        z = i * 2.5
        points.append((x, y, z))
    write_points("helix.xyz", points, 3)


def gen_toroidal_3d():
    # A doubly-curved "eggbox"/toroidal surface boundary path -- one of
    # the deployed shapes named explicitly in the source paper.
    points = []
    R, r = 800.0, 250.0
    n = 5000
    for i in range(n):
        u = i * 0.31   # large-circle angle step
        v = i * 2.05   # small-circle angle step (different rate -> toroidal spiral)
        x = (R + r * math.cos(v)) * math.cos(u)
        y = (R + r * math.cos(v)) * math.sin(u)
        z = r * math.sin(v)
        points.append((x, y, z))
    write_points("toroidal.xyz", points, 3)


def gen_gps_track_2d():
    random.seed(3)
    points = []
    x, y = 0.0, 0.0
    heading = 0.0
    for _ in range(3000):
        heading += random.uniform(-0.05, 0.05)  # smooth, noisy real-world drift
        speed = 8.0 + random.uniform(-0.5, 0.5)
        x += speed * math.cos(heading)
        y += speed * math.sin(heading)
        points.append((x, y))
    write_points("gps_track.xy", points, 2)


def gen_lidar_ring_3d():
    # One rotation of a single spinning-LiDAR ring: ~constant radius,
    # small per-shot range jitter and elevation noise, stepped azimuth.
    random.seed(4)
    points = []
    n = 1800  # 0.2 degree angular resolution
    base_radius = 3000.0  # mm
    for i in range(n):
        az = 2 * math.pi * i / n
        radius = base_radius + random.uniform(-15, 15) + 200 * math.sin(az * 3)
        elev_noise = random.uniform(-5, 5)
        x = radius * math.cos(az)
        y = radius * math.sin(az)
        z = elev_noise
        points.append((x, y, z))
    write_points("lidar_ring.xyz", points, 3)


def gen_random_walk_2d():
    random.seed(5)
    points = []
    x, y = 0.0, 0.0
    for _ in range(3000):
        x += random.uniform(-10, 10)
        y += random.uniform(-10, 10)
        points.append((x, y))
    write_points("random_walk.xy", points, 2)


if __name__ == "__main__":
    gen_text()
    gen_smooth_gradient()
    gen_random()
    gen_spiral_2d()
    gen_helix_3d()
    gen_toroidal_3d()
    gen_gps_track_2d()
    gen_lidar_ring_3d()
    gen_random_walk_2d()
