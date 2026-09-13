#!/usr/bin/env python3
"""Builds bench/datasets_real/source_corpus.bin: a real, non-synthetic
benchmark corpus assembled by concatenating C/C++ source files from a
large local repository, up to a size cap. Used by
real_corpus_benchmark.py.

Not checked into git (see .gitignore) -- it's derived from whatever large
codebase happens to be on this machine, not authored for this repo, and
redistributing someone else's source tree isn't appropriate. Point this
at any sufficiently large local C/C++ project you have:

    python bench/build_real_corpus.py /path/to/some/large/cpp/project
"""
import os
import sys

EXCLUDE_DIRS = {"build", ".git", "node_modules", "cicd", "docker", "cmake-build-debug", "cmake-build-release"}
MAX_BYTES = 20_000_000
EXTENSIONS = (".cpp", ".h", ".hpp", ".cc", ".c")


def main():
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} <path-to-large-cpp-project>", file=sys.stderr)
        sys.exit(1)
    root = sys.argv[1]
    if not os.path.isdir(root):
        print(f"error: {root} is not a directory", file=sys.stderr)
        sys.exit(1)

    out = bytearray()
    count = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        for fn in filenames:
            if fn.endswith(EXTENSIONS):
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "rb") as f:
                        data = f.read()
                except OSError:
                    continue
                out.extend(data)
                count += 1
                if len(out) >= MAX_BYTES:
                    break
        if len(out) >= MAX_BYTES:
            break

    out = out[:MAX_BYTES]
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets_real")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "source_corpus.bin")
    with open(out_path, "wb") as f:
        f.write(bytes(out))
    print(f"wrote {out_path}: {count} files, {len(out):,} bytes")


if __name__ == "__main__":
    main()
