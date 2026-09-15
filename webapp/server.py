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

Run:
    pip install -r requirements.txt
    python server.py [--scissorc PATH] [--port 8000]
Then open http://127.0.0.1:8000/ in a browser.

Not deployed anywhere -- this only runs locally until a hosting decision
is made deliberately (see DESIGN.md's "Local web app" section).
"""
import argparse
import json
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
        resp = Response(data, mimetype="application/octet-stream")
        resp.headers["Content-Disposition"] = f'attachment; filename="{download_name}"'
        resp.headers["X-Csa-Report"] = json.dumps(report)
        resp.headers["Access-Control-Expose-Headers"] = "X-Csa-Report, Content-Disposition"
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
