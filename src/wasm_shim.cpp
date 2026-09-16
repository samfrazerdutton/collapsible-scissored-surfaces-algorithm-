// src/wasm_shim.cpp -- a thin, WASM-specific re-export of csa_capi.h's
// functions. Not part of the native build (see CMakeLists.txt: only
// compiled by bench/build_wasm.sh, never by the CMake project).
//
// Emscripten lowers a struct-by-value return (csa_buffer) into a hidden
// pointer parameter under the hood, which JS's cwrap/ccall have no way
// to know about -- so every wrapped call here takes an explicit
// out-param pointer for the size instead and returns just the raw data
// pointer, which is what a JS caller can actually consume directly
// (Module.HEAPU8.subarray(ptr, ptr + size)).
#include "csa/csa_capi.h"
#include "csa/spatial_index.hpp"
#include <algorithm>
#include <cstdlib>
#include <emscripten/emscripten.h>

// A single, page-global KdTree3i rather than an opaque per-tree handle:
// the browser demo this shim serves decodes and queries one point cloud
// at a time (see docs/index.html), so there is nothing for a caller to
// gain from managing multiple trees, and a lot of JS-side handle-
// lifetime bookkeeping to avoid by not offering that. build() replaces
// whatever tree existed before, exactly like reloading a new file
// replaces whatever was previously decoded.
static csa::KdTree3i g_kdtree;

extern "C" {

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress(const unsigned char* input, size_t input_size, size_t* out_size) {
    csa_buffer b = csa_compress(input, input_size, /*use_gpu=*/0);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_decompress(const unsigned char* input, size_t input_size, size_t* out_size) {
    csa_buffer b = csa_decompress(input, input_size);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_pose(const int32_t* pose7, size_t count, size_t* out_size) {
    csa_buffer b = csa_compress_pose(pose7, count);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_pose_lossy(const int32_t* pose7, size_t count,
                                             uint32_t pos_quant, uint32_t pos_resync,
                                             uint32_t quat_quant, uint32_t quat_resync,
                                             size_t* out_size) {
    csa_buffer b = csa_compress_pose_lossy(pose7, count, pos_quant, pos_resync, quat_quant, quat_resync);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
int32_t* csa_wasm_decompress_pose(const unsigned char* input, size_t input_size, size_t* out_count) {
    csa_buffer b = csa_decompress_pose(input, input_size, out_count);
    return (int32_t*)b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_geo2d(const int32_t* xy, size_t count, size_t* out_size) {
    csa_buffer b = csa_compress_geo2d(xy, count);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_geo2d_lossy(const int32_t* xy, size_t count,
                                              uint32_t quant_step, uint32_t resync_interval,
                                              size_t* out_size) {
    csa_buffer b = csa_compress_geo2d_lossy(xy, count, quant_step, resync_interval);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
int32_t* csa_wasm_decompress_geo2d(const unsigned char* input, size_t input_size, size_t* out_count) {
    csa_buffer b = csa_decompress_geo2d(input, input_size, out_count);
    return (int32_t*)b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_geo3d(const int32_t* xyz, size_t count, size_t* out_size) {
    csa_buffer b = csa_compress_geo3d(xyz, count);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_compress_geo3d_lossy(const int32_t* xyz, size_t count,
                                              uint32_t quant_step, uint32_t resync_interval,
                                              size_t* out_size) {
    csa_buffer b = csa_compress_geo3d_lossy(xyz, count, quant_step, resync_interval);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
int32_t* csa_wasm_decompress_geo3d(const unsigned char* input, size_t input_size, size_t* out_count) {
    csa_buffer b = csa_decompress_geo3d(input, input_size, out_count);
    return (int32_t*)b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_rans_encode(const unsigned char* input, size_t input_size,
                                     int num_lanes, int scale_bits, size_t* out_size) {
    csa_buffer b = csa_rans_encode(input, input_size, num_lanes, scale_bits);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
unsigned char* csa_wasm_rans_decode(const unsigned char* input, size_t input_size, size_t* out_size) {
    csa_buffer b = csa_rans_decode(input, input_size);
    *out_size = b.size;
    return b.data;
}

EMSCRIPTEN_KEEPALIVE
void csa_wasm_kdtree_build(const int32_t* xyz, size_t count) {
    std::vector<csa::Point3i> pts(count);
    for (size_t i = 0; i < count; i++)
        pts[i] = {xyz[i * 3 + 0], xyz[i * 3 + 1], xyz[i * 3 + 2]};
    g_kdtree.build(pts);
}

// Both query functions return a malloc'd uint32_t array of original
// point indices (free with csa_wasm_free, the same generic free()
// wrapper every other allocating export here already uses) and write
// the result count through out_count. An empty/never-built tree simply
// returns zero results, not an error -- there's nothing unsafe about
// querying before build() is called, it just finds nothing.
EMSCRIPTEN_KEEPALIVE
uint32_t* csa_wasm_kdtree_range_query(const int32_t* lo3, const int32_t* hi3, size_t* out_count) {
    csa::Point3i lo{lo3[0], lo3[1], lo3[2]};
    csa::Point3i hi{hi3[0], hi3[1], hi3[2]};
    std::vector<uint32_t> result = g_kdtree.range_query(lo, hi);
    *out_count = result.size();
    uint32_t* out = (uint32_t*)std::malloc(result.size() * sizeof(uint32_t));
    if (out) std::copy(result.begin(), result.end(), out);
    return out;
}

EMSCRIPTEN_KEEPALIVE
uint32_t* csa_wasm_kdtree_knn(const int32_t* query3, size_t k, size_t* out_count) {
    csa::Point3i q{query3[0], query3[1], query3[2]};
    std::vector<uint32_t> result = g_kdtree.k_nearest(q, k);
    *out_count = result.size();
    uint32_t* out = (uint32_t*)std::malloc(result.size() * sizeof(uint32_t));
    if (out) std::copy(result.begin(), result.end(), out);
    return out;
}

EMSCRIPTEN_KEEPALIVE
void csa_wasm_free(void* ptr) {
    std::free(ptr);
}

EMSCRIPTEN_KEEPALIVE
const char* csa_wasm_last_error(void) {
    return csa_last_error();
}

} // extern "C"
