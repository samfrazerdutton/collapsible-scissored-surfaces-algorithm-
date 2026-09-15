/* csa_capi.h -- stable C ABI for libcsa.
 *
 * Pure C (no C++ types cross this boundary), so it links the same way
 * regardless of which compiler/toolchain built libcsa vs. which one is
 * calling it (MSVC, GCC, Clang) -- the classic reason for a C-ABI shim
 * around a C++ core. This is what every language binding (Python ctypes,
 * Rust FFI, C#/Go via cgo, ...) should be built against, instead of
 * talking to the C++ headers directly.
 */
#ifndef CSA_CAPI_H
#define CSA_CAPI_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
  #if defined(CSA_BUILDING_DLL)
    #define CSA_API __declspec(dllexport)
  #else
    #define CSA_API __declspec(dllimport)
  #endif
#else
  #define CSA_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* A heap-allocated output buffer. `data` is NULL and `size` is 0 both for
 * a genuinely empty result and for a failed call (check csa_last_error()
 * to tell them apart). Always release with csa_free_buffer(). */
typedef struct csa_buffer {
    unsigned char* data;
    size_t size;
} csa_buffer;

/* General-purpose byte-stream compression: tries raw storage, the
 * Pantograph Lift, and the LZ dictionary matcher, and keeps whichever
 * encodes smallest. use_gpu != 0 tries the CUDA path for the Pantograph
 * Lift candidate if a device is available, falling back transparently
 * otherwise -- the bitstream is identical either way. */
CSA_API csa_buffer csa_compress(const unsigned char* input, size_t input_size, int use_gpu);
CSA_API csa_buffer csa_decompress(const unsigned char* input, size_t input_size);

/* Geo2D: points passed/returned as interleaved int32 (x0,y0,x1,y1,...).
 * `count` is the number of points (i.e. `xy` has 2*count elements). */
CSA_API csa_buffer csa_compress_geo2d(const int32_t* xy, size_t count);
/* quant_step <= 1 is lossless (identical to csa_compress_geo2d);
 * resync_interval == 0 means no periodic exact resync. */
CSA_API csa_buffer csa_compress_geo2d_lossy(const int32_t* xy, size_t count,
                                             uint32_t quant_step, uint32_t resync_interval);
/* Returns a buffer of interleaved int32 (x,y) pairs; *out_count receives
 * the number of points. Works for both lossless and lossy blobs -- the
 * quantization parameters ride in the blob itself. */
CSA_API csa_buffer csa_decompress_geo2d(const unsigned char* input, size_t input_size, size_t* out_count);

/* Geo3D: points as interleaved int32 (x0,y0,z0,x1,y1,z1,...). */
CSA_API csa_buffer csa_compress_geo3d(const int32_t* xyz, size_t count);
/* Same lossy design as csa_compress_geo2d_lossy. Tries both the xy+z
 * composition and the true 3D similarity joint and keeps whichever
 * encodes smaller (see codec.hpp's compress_geo3d_lossy doc comment).
 * quant_step <= 1 is lossless (identical to csa_compress_geo3d). */
CSA_API csa_buffer csa_compress_geo3d_lossy(const int32_t* xyz, size_t count,
                                             uint32_t quant_step, uint32_t resync_interval);
CSA_API csa_buffer csa_decompress_geo3d(const unsigned char* input, size_t input_size, size_t* out_count);

/* Pose: 6-DOF samples (position + unit quaternion orientation) as
 * interleaved int32, 7 per pose (x0,y0,z0,qw0,qx0,qy0,qz0,x1,...) --
 * `count` is the number of poses, so `pose7` has 7*count elements.
 * Position and orientation are compressed independently (unrelated
 * structure) via the Geo3D auto-select and the Quaternion Joint
 * respectively -- see quaternion_joint.hpp/DESIGN.md. */
CSA_API csa_buffer csa_compress_pose(const int32_t* pose7, size_t count);
/* pos_quant_step/pos_resync_interval reach the position half's existing
 * lossy support; quat_quant_step/quat_resync_interval are the analogous
 * knobs for the Quaternion Joint. Either quant_step <= 1 is lossless for
 * that half; both <= 1 is identical to csa_compress_pose. */
CSA_API csa_buffer csa_compress_pose_lossy(const int32_t* pose7, size_t count,
                                            uint32_t pos_quant_step, uint32_t pos_resync_interval,
                                            uint32_t quat_quant_step, uint32_t quat_resync_interval);
/* Returns a buffer of interleaved int32 pose7 tuples; *out_count receives
 * the number of poses. Works for both lossless and lossy blobs. */
CSA_API csa_buffer csa_decompress_pose(const unsigned char* input, size_t input_size, size_t* out_count);

/* Interleaved rANS: a parallel, order-0 static-table entropy coder (see
 * csa/rans_coder.hpp) offered here as a standalone alternative to the
 * general-purpose csa_compress -- it is NOT used internally by
 * csa_compress, since it does not consistently beat the adaptive order-1
 * range coder csa_compress already uses (see DESIGN.md for the measured
 * comparison). Reach for this directly when you specifically want
 * multi-threaded CPU encode/decode across independent lanes, e.g. for
 * very large buffers on a multi-core machine.
 * num_lanes <= 0 is treated as 1; scale_bits <= 0 or > 16 defaults to 14. */
CSA_API csa_buffer csa_rans_encode(const unsigned char* input, size_t input_size,
                                    int num_lanes, int scale_bits);
CSA_API csa_buffer csa_rans_decode(const unsigned char* input, size_t input_size);

/* Frees a buffer returned by any csa_compress_... or csa_decompress_... function. */
CSA_API void csa_free_buffer(csa_buffer buf);

/* Human-readable description of the most recent failure on this thread
 * (empty string if the last call on this thread succeeded). The pointer
 * is valid until the next csa_* call on the same thread. */
CSA_API const char* csa_last_error(void);

/* 1 if a CUDA device is available at runtime, 0 otherwise. */
CSA_API int csa_cuda_available(void);

#ifdef __cplusplus
}
#endif

#endif /* CSA_CAPI_H */
