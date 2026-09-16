#!/bin/bash
# Builds the WASM module src/wasm_shim.cpp exports (compress/decompress,
# rANS, and the KdTree3i spatial-index queries) into a single
# self-contained JS file (SINGLE_FILE=1 inlines the wasm binary as base64
# directly in the JS -- no separate .wasm to ship or fetch).
#
# Requires the Emscripten SDK (emsdk) activated in the current shell
# (`source /path/to/emsdk/emsdk_env.sh` first) -- this script does not
# activate it for you, since where emsdk lives is a per-machine detail.
#
# This file was itself a real, disclosed documentation gap until now:
# src/wasm_shim.cpp's own header comment referenced it by name for a
# long time before it actually existed, because the module had only ever
# been rebuilt by hand, ad hoc, without saving the exact command used --
# the same "referenced but not yet created" gap this project already
# found and fixed once for docs/SANITIZERS.md. This script is that fix
# for the WASM build: the exact command, verified (see
# build_wasm/test_kdtree.mjs -- not committed, but the verification it
# ran is described in DESIGN.md's spatial-index section) to produce a
# module where every export, old and new, round-trips correctly.
set -e
cd "$(dirname "$0")/.."

SOURCES="src/bwt_codec.cpp src/bwt_transform.cpp src/codec.cpp src/crc32.cpp \
src/lz_codec.cpp src/lz_matcher.cpp src/packet_transport.cpp \
src/pantograph_lift.cpp src/pantograph_lift_cuda_stub.cpp src/pose_stream.cpp \
src/quaternion_joint.cpp src/range_coder.cpp src/rans_coder.cpp \
src/rod_joint_transform.cpp src/simd.cpp src/simd_cuda_stub.cpp \
src/spatial_index.cpp src/csa_capi.cpp src/wasm_shim.cpp"
# Deliberately excludes src/simd_avx2.cpp (x86-only intrinsics, meaningless
# for the wasm32 target -- simd.hpp's CSA_X86_SIMD guard already makes
# simd.cpp itself fall back to scalar when built for wasm32, so nothing
# else needs to change) and every real CUDA .cu file (pantograph_lift_cuda_stub.cpp
# supplies the CPU-fallback bodies for both the Pantograph Lift and
# quaternion-calibration CUDA entry points, and simd_cuda_stub.cpp for
# max_abs_diff_i32_cuda -- there is no GPU in a browser tab).

mkdir -p build_wasm
em++ $SOURCES -std=c++17 -Iinclude -O2 \
    -s MODULARIZE=1 -s EXPORT_NAME=CsaModule -s SINGLE_FILE=1 \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s EXPORTED_RUNTIME_METHODS='["cwrap","ccall","getValue","HEAP32","HEAPU32","HEAPU8"]' \
    -s EXPORTED_FUNCTIONS='["_malloc","_free"]' \
    -o build_wasm/csa.js

echo "built build_wasm/csa.js"
