// src/csa_capi.cpp -- implementation of the stable C ABI declared in
// csa/csa_capi.h. This file is the only place C++ types (std::vector,
// std::string, csa::Point2i, ...) are allowed to touch the ABI boundary;
// every exported function converts to/from plain C types at its edges.
#include "csa/csa_capi.h"
#include "csa/codec.hpp"
#include "csa/pantograph_lift_cuda.hpp"
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

thread_local std::string g_last_error;

csa_buffer make_buffer(const std::vector<csa::u8>& v) {
    csa_buffer b;
    b.size = v.size();
    b.data = nullptr;
    if (!v.empty()) {
        b.data = (unsigned char*)std::malloc(v.size());
        if (b.data) std::memcpy(b.data, v.data(), v.size());
        else b.size = 0;
    }
    g_last_error.clear();
    return b;
}

csa_buffer make_int32_buffer(const std::vector<int32_t>& flat) {
    csa_buffer b;
    b.size = flat.size() * sizeof(int32_t);
    b.data = nullptr;
    if (!flat.empty()) {
        b.data = (unsigned char*)std::malloc(b.size);
        if (b.data) std::memcpy(b.data, flat.data(), b.size);
        else b.size = 0;
    }
    g_last_error.clear();
    return b;
}

csa_buffer fail(const char* msg) {
    g_last_error = msg;
    csa_buffer b;
    b.data = nullptr;
    b.size = 0;
    return b;
}

} // namespace

extern "C" {

csa_buffer csa_compress(const unsigned char* input, size_t input_size, int use_gpu) {
    try {
        std::vector<csa::u8> in(input, input + input_size);
        return make_buffer(csa::compress(in, use_gpu != 0));
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("csa_compress: unknown error");
    }
}

csa_buffer csa_decompress(const unsigned char* input, size_t input_size) {
    try {
        std::vector<csa::u8> in(input, input + input_size);
        return make_buffer(csa::decompress(in));
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("csa_decompress: unknown error");
    }
}

csa_buffer csa_compress_geo2d(const int32_t* xy, size_t count) {
    try {
        std::vector<csa::Point2i> pts(count);
        for (size_t i = 0; i < count; i++) { pts[i].x = xy[2 * i]; pts[i].y = xy[2 * i + 1]; }
        return make_buffer(csa::compress_geo2d(pts));
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("csa_compress_geo2d: unknown error");
    }
}

csa_buffer csa_compress_geo2d_lossy(const int32_t* xy, size_t count,
                                     uint32_t quant_step, uint32_t resync_interval) {
    try {
        std::vector<csa::Point2i> pts(count);
        for (size_t i = 0; i < count; i++) { pts[i].x = xy[2 * i]; pts[i].y = xy[2 * i + 1]; }
        return make_buffer(csa::compress_geo2d_lossy(pts, quant_step, resync_interval));
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("csa_compress_geo2d_lossy: unknown error");
    }
}

csa_buffer csa_decompress_geo2d(const unsigned char* input, size_t input_size, size_t* out_count) {
    try {
        std::vector<csa::u8> in(input, input + input_size);
        auto pts = csa::decompress_geo2d(in);
        if (out_count) *out_count = pts.size();
        std::vector<int32_t> flat(pts.size() * 2);
        for (size_t i = 0; i < pts.size(); i++) { flat[2 * i] = pts[i].x; flat[2 * i + 1] = pts[i].y; }
        return make_int32_buffer(flat);
    } catch (const std::exception& e) {
        if (out_count) *out_count = 0;
        return fail(e.what());
    } catch (...) {
        if (out_count) *out_count = 0;
        return fail("csa_decompress_geo2d: unknown error");
    }
}

csa_buffer csa_compress_geo3d(const int32_t* xyz, size_t count) {
    try {
        std::vector<csa::Point3i> pts(count);
        for (size_t i = 0; i < count; i++) {
            pts[i].x = xyz[3 * i];
            pts[i].y = xyz[3 * i + 1];
            pts[i].z = xyz[3 * i + 2];
        }
        return make_buffer(csa::compress_geo3d(pts));
    } catch (const std::exception& e) {
        return fail(e.what());
    } catch (...) {
        return fail("csa_compress_geo3d: unknown error");
    }
}

csa_buffer csa_decompress_geo3d(const unsigned char* input, size_t input_size, size_t* out_count) {
    try {
        std::vector<csa::u8> in(input, input + input_size);
        auto pts = csa::decompress_geo3d(in);
        if (out_count) *out_count = pts.size();
        std::vector<int32_t> flat(pts.size() * 3);
        for (size_t i = 0; i < pts.size(); i++) {
            flat[3 * i] = pts[i].x;
            flat[3 * i + 1] = pts[i].y;
            flat[3 * i + 2] = pts[i].z;
        }
        return make_int32_buffer(flat);
    } catch (const std::exception& e) {
        if (out_count) *out_count = 0;
        return fail(e.what());
    } catch (...) {
        if (out_count) *out_count = 0;
        return fail("csa_decompress_geo3d: unknown error");
    }
}

void csa_free_buffer(csa_buffer buf) {
    if (buf.data) std::free(buf.data);
}

const char* csa_last_error(void) {
    return g_last_error.c_str();
}

int csa_cuda_available(void) {
    return csa::cuda_is_available() ? 1 : 0;
}

} // extern "C"
