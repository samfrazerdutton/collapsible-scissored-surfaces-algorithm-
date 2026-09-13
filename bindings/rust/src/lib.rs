//! Rust bindings for libcsa (the Collapsible Scissored Surfaces Algorithm),
//! via FFI on top of the stable C ABI in `include/csa/csa_capi.h` -- the
//! same ABI `bindings/python/csa.py` is built on, so the two should always
//! agree on behavior.
//!
//! ```no_run
//! # fn main() -> Result<(), csa::CsaError> {
//! let compressed = csa::compress(b"hello world", false)?;
//! let original = csa::decompress(&compressed)?;
//! assert_eq!(original, b"hello world");
//!
//! let points = vec![(0, 0), (3, 1), (7, 2)];
//! let blob = csa::compress_geo2d(&points)?;
//! let back = csa::decompress_geo2d(&blob)?;
//! assert_eq!(back, points);
//!
//! // quant_step <= 1 is lossless; resync_interval bounds absolute drift.
//! let lossy = csa::compress_geo2d_lossy(&points, 20, 64)?;
//! let _ = csa::decompress_geo2d(&lossy)?; // decode is the same function
//! # Ok(())
//! # }
//! ```
use std::error::Error;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::{c_char, c_int, c_uchar};

#[repr(C)]
struct CsaBuffer {
    data: *mut c_uchar,
    size: usize,
}

extern "C" {
    fn csa_compress(input: *const c_uchar, input_size: usize, use_gpu: c_int) -> CsaBuffer;
    fn csa_decompress(input: *const c_uchar, input_size: usize) -> CsaBuffer;

    fn csa_compress_geo2d(xy: *const i32, count: usize) -> CsaBuffer;
    fn csa_compress_geo2d_lossy(
        xy: *const i32,
        count: usize,
        quant_step: u32,
        resync_interval: u32,
    ) -> CsaBuffer;
    fn csa_decompress_geo2d(
        input: *const c_uchar,
        input_size: usize,
        out_count: *mut usize,
    ) -> CsaBuffer;

    fn csa_compress_geo3d(xyz: *const i32, count: usize) -> CsaBuffer;
    fn csa_compress_geo3d_lossy(
        xyz: *const i32,
        count: usize,
        quant_step: u32,
        resync_interval: u32,
    ) -> CsaBuffer;
    fn csa_decompress_geo3d(
        input: *const c_uchar,
        input_size: usize,
        out_count: *mut usize,
    ) -> CsaBuffer;

    fn csa_free_buffer(buf: CsaBuffer);
    fn csa_last_error() -> *const c_char;
    fn csa_cuda_available() -> c_int;
}

/// An error reported by libcsa, sourced from `csa_last_error()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CsaError(pub String);

impl fmt::Display for CsaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl Error for CsaError {}

// A genuinely empty result (e.g. decompressing an empty file) looks
// identical to a failed call at the struct level (data == null, size ==
// 0) -- csa_last_error() disambiguates, and it's cleared on every success,
// same reasoning as the Python bindings' _check_and_extract.
unsafe fn check_and_extract(buf: CsaBuffer) -> Result<Vec<u8>, CsaError> {
    if buf.data.is_null() && buf.size == 0 {
        let err_ptr = csa_last_error();
        if !err_ptr.is_null() {
            let msg = CStr::from_ptr(err_ptr).to_string_lossy().into_owned();
            if !msg.is_empty() {
                return Err(CsaError(msg));
            }
        }
        return Ok(Vec::new());
    }
    let slice = std::slice::from_raw_parts(buf.data, buf.size);
    let out = slice.to_vec();
    csa_free_buffer(buf);
    Ok(out)
}

// The C ABI hands back a plain malloc'd byte buffer; reinterpreting it as
// i32 without an alignment guarantee from the allocator would be
// undefined behavior on a strict-alignment read, so every multi-byte
// field is read with `read_unaligned` instead of casting-and-dereferencing.
unsafe fn read_i32_at(bytes: &[u8], byte_offset: usize) -> i32 {
    (bytes.as_ptr().add(byte_offset) as *const i32).read_unaligned()
}

/// General-purpose compression: tries raw storage, the Pantograph Lift,
/// and the LZ dictionary matcher, and keeps whichever encodes smallest.
/// `use_gpu` tries the CUDA path for the Pantograph Lift candidate if a
/// device is available, falling back transparently otherwise.
pub fn compress(data: &[u8], use_gpu: bool) -> Result<Vec<u8>, CsaError> {
    unsafe {
        let buf = csa_compress(data.as_ptr(), data.len(), if use_gpu { 1 } else { 0 });
        check_and_extract(buf)
    }
}

pub fn decompress(data: &[u8]) -> Result<Vec<u8>, CsaError> {
    unsafe { check_and_extract(csa_decompress(data.as_ptr(), data.len())) }
}

fn points2d_to_flat(points: &[(i32, i32)]) -> Vec<i32> {
    let mut v = Vec::with_capacity(points.len() * 2);
    for &(x, y) in points {
        v.push(x);
        v.push(y);
    }
    v
}

fn points3d_to_flat(points: &[(i32, i32, i32)]) -> Vec<i32> {
    let mut v = Vec::with_capacity(points.len() * 3);
    for &(x, y, z) in points {
        v.push(x);
        v.push(y);
        v.push(z);
    }
    v
}

/// `points`: 2D integer coordinate pairs.
pub fn compress_geo2d(points: &[(i32, i32)]) -> Result<Vec<u8>, CsaError> {
    let flat = points2d_to_flat(points);
    unsafe { check_and_extract(csa_compress_geo2d(flat.as_ptr(), points.len())) }
}

/// `quant_step <= 1` is lossless (identical to [`compress_geo2d`]).
/// `resync_interval` periodically stores an exact rod to bound how far
/// absolute position error can drift (0 = never); see DESIGN.md.
pub fn compress_geo2d_lossy(
    points: &[(i32, i32)],
    quant_step: u32,
    resync_interval: u32,
) -> Result<Vec<u8>, CsaError> {
    let flat = points2d_to_flat(points);
    unsafe {
        check_and_extract(csa_compress_geo2d_lossy(
            flat.as_ptr(),
            points.len(),
            quant_step,
            resync_interval,
        ))
    }
}

/// Works for both lossless and lossy blobs -- the quantization parameters
/// ride in the blob itself, there is no separate lossy decode entry point.
pub fn decompress_geo2d(data: &[u8]) -> Result<Vec<(i32, i32)>, CsaError> {
    let mut out_count: usize = 0;
    unsafe {
        let buf = csa_decompress_geo2d(data.as_ptr(), data.len(), &mut out_count as *mut usize);
        let raw = check_and_extract(buf)?;
        let n = out_count;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let x = read_i32_at(&raw, i * 8);
            let y = read_i32_at(&raw, i * 8 + 4);
            result.push((x, y));
        }
        Ok(result)
    }
}

/// `points`: 3D integer coordinate triples.
pub fn compress_geo3d(points: &[(i32, i32, i32)]) -> Result<Vec<u8>, CsaError> {
    let flat = points3d_to_flat(points);
    unsafe { check_and_extract(csa_compress_geo3d(flat.as_ptr(), points.len())) }
}

/// Same design as [`compress_geo2d_lossy`], on the true 3D similarity
/// joint. `quant_step <= 1` is lossless (identical to [`compress_geo3d`]).
pub fn compress_geo3d_lossy(
    points: &[(i32, i32, i32)],
    quant_step: u32,
    resync_interval: u32,
) -> Result<Vec<u8>, CsaError> {
    let flat = points3d_to_flat(points);
    unsafe {
        check_and_extract(csa_compress_geo3d_lossy(
            flat.as_ptr(),
            points.len(),
            quant_step,
            resync_interval,
        ))
    }
}

pub fn decompress_geo3d(data: &[u8]) -> Result<Vec<(i32, i32, i32)>, CsaError> {
    let mut out_count: usize = 0;
    unsafe {
        let buf = csa_decompress_geo3d(data.as_ptr(), data.len(), &mut out_count as *mut usize);
        let raw = check_and_extract(buf)?;
        let n = out_count;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let x = read_i32_at(&raw, i * 12);
            let y = read_i32_at(&raw, i * 12 + 4);
            let z = read_i32_at(&raw, i * 12 + 8);
            result.push((x, y, z));
        }
        Ok(result)
    }
}

pub fn cuda_available() -> bool {
    unsafe { csa_cuda_available() != 0 }
}
