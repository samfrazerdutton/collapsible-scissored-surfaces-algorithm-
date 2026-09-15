#!/usr/bin/env python3
"""A real, runnable web front end for scissorc's squeeze/unsqueeze commands
-- the actual local "app" this project didn't have before: drag your own
file in, get a real compressed file back, in a real (non-sandboxed)
browser, unlike the WASM demo (demo/csa_demo.html), which can prove the
codec works on sample data but cannot hand a file back to whoever's
looking at it.

This is deliberately a thin transport layer, not a second brain: all the
"smart" logic (sniffing a file's shape, picking a lossless scale, running
the real round-trip check) lives in exactly one place, cli/main.cpp's
squeeze/unsqueeze commands, already tested against real files. This
server just shells out to the already-built scissorc binary and relays
its own stdout report and output file back to the browser -- so there is
no second implementation of that logic anywhere to drift out of sync.

Two things this file adds on top of that report, because a bare
percentage number with nothing to compare it to isn't actually
informative, and this codec's real edge (structured position/orientation
data) is invisible if the first file anyone tries is a PDF:
  - a live gzip -9 size on the same input, computed with the stdlib's own
    zlib, so "smaller" has a real baseline instead of just a percentage;
  - synthetic sample datasets (a pose trajectory, a point cloud, a GPS
    track) generated on request, so the first thing someone can click
    shows the domain this codec is actually built for -- see
    DESIGN.md/REAL_POSE_BENCHMARK.md for why that's the honest framing:
    general files like a PDF are NOT this codec's specialty, and pretending
    otherwise with a good number on the wrong kind of file is misleading.

Run:
    pip install -r requirements.txt
    python server.py [--scissorc PATH] [--port 8000]
Then open http://127.0.0.1:8000/ in a browser.

Not deployed anywhere -- this only runs locally until a hosting decision
is made deliberately (see DESIGN.md's "Local web app" section).
"""
import argparse
import gzip
import json
import math
import os
import subprocess
import tempfile

from flask import Flask, Response, request
from werkzeug.utils import secure_filename

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)

app = Flask(__name__, static_folder="static", static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB -- generous for a local demo, still bounded

SCISSORC = os.environ.get(
    "SCISSORC_PATH",
    os.path.join(REPO_ROOT, "build", "scissorc.exe" if os.name == "nt" else "scissorc"),
)


def run_scissorc(*args, timeout=180):
    return subprocess.run([SCISSORC, *args], capture_output=True, text=True, timeout=timeout)


def parse_report(stdout, in_path, in_name, out_path, out_name):
    # scissorc's own stdout echoes back the exact paths it was given --
    # here those are this request's throwaway temp-directory paths, which
    # would otherwise leak the server's local filesystem layout to the
    # browser for no reason. Swap in the names the user actually knows.
    stdout = stdout.replace(in_path, in_name).replace(out_path, out_name)
    lines = [l for l in stdout.strip().split("\n") if l.strip()]
    return {"summary": lines[0] if lines else "", "detail": lines[1:]}


def detect_shape(detail_lines):
    joined = " ".join(detail_lines)
    for shape in ("geo2d", "geo3d", "pose"):
        if shape in joined:
            return shape
    return "general"


def read_preview_points(path, max_points=2000):
    """Downsamples a squeeze/unsqueeze-restored numeric text file to at
    most max_points (x, y) pairs (first two columns of whatever shape it
    is -- geo2d/geo3d/pose all have position first) for a client-side
    trajectory/scatter plot. Returns [] if the file isn't such a table."""
    try:
        with open(path, "r") as f:
            rows = []
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                try:
                    rows.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    return []
    except OSError:
        return []
    if not rows:
        return []
    stride = max(1, len(rows) // max_points)
    return rows[::stride]


def error_response(message, status=400):
    return Response(json.dumps({"error": message}), status=status, mimetype="application/json")


@app.route("/")
def index():
    return app.send_static_file("index.html")


@app.route("/api/squeeze", methods=["POST"])
def api_squeeze():
    f = request.files.get("file")
    if not f or not f.filename:
        return error_response("no file uploaded")
    quality = request.form.get("quality", "").strip()

    with tempfile.TemporaryDirectory() as td:
        in_name = secure_filename(f.filename) or "upload.bin"
        in_path = os.path.join(td, in_name)
        f.save(in_path)
        out_path = in_path + ".csa"

        with open(in_path, "rb") as raw_f:
            raw_bytes = raw_f.read()
        gzip_size = len(gzip.compress(raw_bytes, compresslevel=9))

        args = ["squeeze", in_path, out_path, "--explain", "--force"]
        if quality:
            args += ["--quality", quality]

        try:
            r = run_scissorc(*args)
        except FileNotFoundError:
            return error_response(f"scissorc binary not found at {SCISSORC} -- build it first (cmake --build build)", 500)
        except subprocess.TimeoutExpired:
            return error_response("compression timed out", 504)

        if r.returncode != 0:
            return error_response(r.stderr.strip() or r.stdout.strip() or "compression failed")

        with open(out_path, "rb") as out_f:
            data = out_f.read()

        report = parse_report(r.stdout, in_path, in_name, out_path, in_name + ".csa")
        report["raw_size"] = len(raw_bytes)
        report["csa_size"] = len(data)
        report["gzip_size"] = gzip_size
        shape = detect_shape(report["detail"])
        report["shape"] = shape

        preview = []
        if shape in ("geo2d", "geo3d", "pose"):
            preview_path = out_path + ".preview"
            pr = run_scissorc("unsqueeze", out_path, preview_path, "--force")
            if pr.returncode == 0:
                preview = read_preview_points(preview_path)
        report["preview"] = preview

        resp = Response(data, mimetype="application/octet-stream")
        resp.headers["Content-Disposition"] = f'attachment; filename="{in_name}.csa"'
        resp.headers["X-Csa-Report"] = json.dumps(report)
        resp.headers["Access-Control-Expose-Headers"] = "X-Csa-Report, Content-Disposition"
        return resp


@app.route("/api/unsqueeze", methods=["POST"])
def api_unsqueeze():
    f = request.files.get("file")
    if not f or not f.filename:
        return error_response("no file uploaded")

    with tempfile.TemporaryDirectory() as td:
        in_name = secure_filename(f.filename) or "upload.csa"
        in_path = os.path.join(td, in_name)
        f.save(in_path)
        out_path = in_path + ".restored"

        args = ["unsqueeze", in_path, out_path, "--explain", "--force"]
        try:
            r = run_scissorc(*args)
        except FileNotFoundError:
            return error_response(f"scissorc binary not found at {SCISSORC} -- build it first (cmake --build build)", 500)
        except subprocess.TimeoutExpired:
            return error_response("decompression timed out", 504)

        if r.returncode != 0:
            return error_response(r.stderr.strip() or r.stdout.strip() or "decompression failed")

        with open(out_path, "rb") as out_f:
            data = out_f.read()

        # Strip a trailing ".csa" from the download name if present, so
        # round-tripping a file named mytrajectory.csv.csa gives back
        # mytrajectory.csv rather than mytrajectory.csv.csa.restored.
        download_name = in_name[:-4] if in_name.lower().endswith(".csa") else in_name + ".restored"

        report = parse_report(r.stdout, in_path, in_name, out_path, download_name)
        report["shape"] = detect_shape(report["detail"])
        report["preview"] = read_preview_points(out_path) if report["shape"] != "general" else []

        resp = Response(data, mimetype="application/octet-stream")
        resp.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        resp.headers["X-Csa-Report"] = json.dumps(report)
        resp.headers["Access-Control-Expose-Headers"] = "X-Csa-Report, Content-Disposition"
        return resp


# ---- sample datasets: the actual point of this codec, one click away ----
# Synthetic, not real recordings (real ones live under bench/_thirdparty/,
# gitignored for size and fetched on demand -- see bench/*.py) -- but the
# same shape class as the real KITTI/EuRoC pose data and the real LiDAR
# ring benchmark already measured elsewhere in this repo, so the ratios
# shown here land in the same neighborhood as the honestly-measured real
# numbers in REAL_POSE_BENCHMARK.md/REAL_GEO_BENCHMARK.md, not inflated.
def make_pose_sample(n=2000):
    lines = []
    qw, qz = 1.0, 0.0
    z = 0.0
    seed = 4242
    def rnd():
        nonlocal seed
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        return (seed >> 8) / float(1 << 24)
    for i in range(n):
        t = i * 0.05
        x = 2000 * math.cos(t)
        y = 2000 * math.sin(t)
        z += 4.0 + (rnd() - 0.5) * 0.5
        lines.append(f"{x:.6f} {y:.6f} {z:.6f} {qw:.6f} 0.000000 0.000000 {qz:.6f}")
        deg = 2.0 + (rnd() - 0.5) * 0.3
        half = math.radians(deg) / 2
        dqw, dqz = math.cos(half), math.sin(half)
        qw, qz = qw * dqw - qz * dqz, qw * dqz + qz * dqw
    return ("\n".join(lines) + "\n").encode(), "sample_drone_pose_trajectory.txt"


def make_points_sample(n=1800):
    lines = []
    for i in range(n):
        a = i * (2 * math.pi / 60)
        r = 3000 - (i / n) * 8
        x = r * math.cos(a)
        y = r * math.sin(a)
        z = -4.0 + 0.5 * math.sin(a * 3)
        lines.append(f"{x:.6f} {y:.6f} {z:.6f}")
    return ("\n".join(lines) + "\n").encode(), "sample_lidar_ring_scan.txt"


def make_gps_sample(n=1500):
    lines = []
    lat, lon = 37.7749, -122.4194
    seed = 99
    def rnd():
        nonlocal seed
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF
        return (seed / float(0x7FFFFFFF)) - 0.5
    heading = 0.0
    for _ in range(n):
        heading += rnd() * 0.15
        lat += math.cos(heading) * 0.00003
        lon += math.sin(heading) * 0.00003
        lines.append(f"{lat:.6f} {lon:.6f}")
    return ("\n".join(lines) + "\n").encode(), "sample_gps_walk.txt"


SAMPLES = {"pose": make_pose_sample, "points": make_points_sample, "gps": make_gps_sample}


@app.route("/api/sample/<name>")
def api_sample(name):
    gen = SAMPLES.get(name)
    if not gen:
        return error_response(f"unknown sample: {name}", 404)
    data, filename = gen()
    resp = Response(data, mimetype="text/plain")
    resp.headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return resp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--scissorc", default=None, help="path to the scissorc binary")
    args = parser.parse_args()
    if args.scissorc:
        SCISSORC = args.scissorc
    print(f"scissorc binary: {SCISSORC}")
    print(f"open http://127.0.0.1:{args.port}/ in a browser")
    app.run(host="127.0.0.1", port=args.port, debug=False)
