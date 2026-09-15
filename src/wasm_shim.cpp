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
#include <cstdlib>
#include <emscripten/emscripten.h>

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
void csa_wasm_free(void* ptr) {
    std::free(ptr);
}

EMSCRIPTEN_KEEPALIVE
const char* csa_wasm_last_error(void) {
    return csa_last_error();
}

} // extern "C"
