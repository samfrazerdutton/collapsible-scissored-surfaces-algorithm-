"""Python bindings for libcsa (the Collapsible Scissored Surfaces Algorithm),
via ctypes on top of the stable C ABI in include/csa/csa_capi.h.

    import csa
    compressed = csa.compress(data)          # bytes -> bytes
    original = csa.decompress(compressed)    # bytes -> bytes

    blob = csa.compress_geo2d(points)              # [(x, y), ...] -> bytes
    points_back = csa.decompress_geo2d(blob)       # bytes -> [(x, y), ...]

    lossy_blob = csa.compress_geo2d_lossy(points, quant_step=20, resync_interval=64)
    # decompress_geo2d works on lossy blobs too -- the quantization
    # parameters ride in the blob itself, there is no separate "lossy mode"
    # decode function. compress_geo3d_lossy/decompress_geo3d are the 3D
    # equivalent.

By default this looks for the built shared library next to this repo's
build/ directory (../../build/csa.dll, libcsa.so, or libcsa.dylib relative
to this file); set the CSA_LIB_PATH environment variable to override that
with an exact path if you're loading it from somewhere else.
"""
import ctypes
import ctypes.util
import os
import platform
import struct

__all__ = [
    "CsaError", "compress", "decompress",
    "compress_geo2d", "compress_geo2d_lossy", "decompress_geo2d",
    "compress_geo3d", "compress_geo3d_lossy", "decompress_geo3d",
    "compress_pose", "compress_pose_lossy", "decompress_pose",
    "cuda_available",
]


class CsaError(Exception):
    """Raised when a libcsa call fails; message comes from csa_last_error()."""


def _find_library() -> str:
    env = os.environ.get("CSA_LIB_PATH")
    if env:
        if os.path.exists(env):
            return env
        raise OSError(f"CSA_LIB_PATH is set to {env!r} but that file does not exist")

    here = os.path.dirname(os.path.abspath(__file__))
    system = platform.system()
    if system == "Windows":
        names = ["csa.dll"]
    elif system == "Darwin":
        names = ["libcsa.dylib"]
    else:
        names = ["libcsa.so"]

    for name in names:
        candidate = os.path.abspath(os.path.join(here, "..", "..", "build", name))
        if os.path.exists(candidate):
            return candidate

    found = ctypes.util.find_library("csa")
    if found:
        return found

    raise OSError(
        "Could not locate the libcsa shared library. Build the project first "
        "(cmake --build build from the repo root), or set the CSA_LIB_PATH "
        "environment variable to the exact .dll/.so/.dylib path."
    )


_lib = ctypes.CDLL(_find_library())


class _CsaBuffer(ctypes.Structure):
    _fields_ = [("data", ctypes.POINTER(ctypes.c_ubyte)), ("size", ctypes.c_size_t)]


_lib.csa_compress.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
_lib.csa_compress.restype = _CsaBuffer
_lib.csa_decompress.argtypes = [ctypes.c_char_p, ctypes.c_size_t]
_lib.csa_decompress.restype = _CsaBuffer
_lib.csa_compress_geo2d.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
_lib.csa_compress_geo2d.restype = _CsaBuffer
_lib.csa_compress_geo2d_lossy.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
]
_lib.csa_compress_geo2d_lossy.restype = _CsaBuffer
_lib.csa_decompress_geo2d.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.csa_decompress_geo2d.restype = _CsaBuffer
_lib.csa_compress_geo3d.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
_lib.csa_compress_geo3d.restype = _CsaBuffer
_lib.csa_compress_geo3d_lossy.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t, ctypes.c_uint32, ctypes.c_uint32,
]
_lib.csa_compress_geo3d_lossy.restype = _CsaBuffer
_lib.csa_decompress_geo3d.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.csa_decompress_geo3d.restype = _CsaBuffer
_lib.csa_compress_pose.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
_lib.csa_compress_pose.restype = _CsaBuffer
_lib.csa_compress_pose_lossy.argtypes = [
    ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t,
    ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
]
_lib.csa_compress_pose_lossy.restype = _CsaBuffer
_lib.csa_decompress_pose.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_lib.csa_decompress_pose.restype = _CsaBuffer
_lib.csa_free_buffer.argtypes = [_CsaBuffer]
_lib.csa_free_buffer.restype = None
_lib.csa_last_error.argtypes = []
_lib.csa_last_error.restype = ctypes.c_char_p
_lib.csa_cuda_available.argtypes = []
_lib.csa_cuda_available.restype = ctypes.c_int


def _check_and_extract(buf: _CsaBuffer) -> bytes:
    if not buf.data and buf.size == 0:
        # Ambiguous at the struct level: a genuinely empty result (e.g.
        # decompressing an empty file) looks identical to a failed call.
        # csa_last_error() disambiguates -- it's cleared on every success.
        err = _lib.csa_last_error()
        if err:
            msg = err.decode("utf-8", "replace")
            if msg:
                raise CsaError(msg)
        return b""
    result = ctypes.string_at(buf.data, buf.size)
    _lib.csa_free_buffer(buf)
    return result


def compress(data: bytes, use_gpu: bool = False) -> bytes:
    """General-purpose compression: tries raw storage, Pantograph Lift, and
    the LZ dictionary matcher, and keeps whichever is smallest."""
    buf = _lib.csa_compress(data, len(data), 1 if use_gpu else 0)
    return _check_and_extract(buf)


def decompress(data: bytes) -> bytes:
    buf = _lib.csa_decompress(data, len(data))
    return _check_and_extract(buf)


def _points2d_to_array(points):
    n = len(points)
    arr = (ctypes.c_int32 * (n * 2))()
    for i, (x, y) in enumerate(points):
        arr[2 * i] = int(x)
        arr[2 * i + 1] = int(y)
    return arr, n


def compress_geo2d(points) -> bytes:
    """points: an iterable of (x, y) integer-ish coordinate pairs."""
    arr, n = _points2d_to_array(points)
    buf = _lib.csa_compress_geo2d(arr, n)
    return _check_and_extract(buf)


def compress_geo2d_lossy(points, quant_step: int, resync_interval: int = 0) -> bytes:
    """quant_step <= 1 is lossless. resync_interval periodically stores an
    exact rod to bound how far absolute position error can drift (0 =
    never); see DESIGN.md for the mechanism."""
    arr, n = _points2d_to_array(points)
    buf = _lib.csa_compress_geo2d_lossy(arr, n, quant_step, resync_interval)
    return _check_and_extract(buf)


def decompress_geo2d(data: bytes):
    """Returns a list of (x, y) tuples. Works for both lossless and lossy
    blobs -- there is no separate lossy decode entry point."""
    out_count = ctypes.c_size_t(0)
    buf = _lib.csa_decompress_geo2d(data, len(data), ctypes.byref(out_count))
    raw = _check_and_extract(buf)
    n = out_count.value
    if n == 0:
        return []
    values = struct.unpack(f"={2 * n}i", raw)
    return [(values[2 * i], values[2 * i + 1]) for i in range(n)]


def _points3d_to_array(points):
    n = len(points)
    arr = (ctypes.c_int32 * (n * 3))()
    for i, (x, y, z) in enumerate(points):
        arr[3 * i] = int(x)
        arr[3 * i + 1] = int(y)
        arr[3 * i + 2] = int(z)
    return arr, n


def compress_geo3d(points) -> bytes:
    """points: an iterable of (x, y, z) integer-ish coordinate triples."""
    arr, n = _points3d_to_array(points)
    buf = _lib.csa_compress_geo3d(arr, n)
    return _check_and_extract(buf)


def compress_geo3d_lossy(points, quant_step: int, resync_interval: int = 0) -> bytes:
    """Same design as compress_geo2d_lossy. Tries both the xy+z composition
    and the true 3D similarity joint and keeps whichever encodes smaller.
    quant_step <= 1 is lossless (identical to compress_geo3d)."""
    arr, n = _points3d_to_array(points)
    buf = _lib.csa_compress_geo3d_lossy(arr, n, quant_step, resync_interval)
    return _check_and_extract(buf)


def decompress_geo3d(data: bytes):
    out_count = ctypes.c_size_t(0)
    buf = _lib.csa_decompress_geo3d(data, len(data), ctypes.byref(out_count))
    raw = _check_and_extract(buf)
    n = out_count.value
    if n == 0:
        return []
    values = struct.unpack(f"={3 * n}i", raw)
    return [(values[3 * i], values[3 * i + 1], values[3 * i + 2]) for i in range(n)]


def _poses_to_array(poses):
    """poses: an iterable of ((x, y, z), (qw, qx, qy, qz)) pairs -- position
    and a unit-quaternion orientation, both integer-ish."""
    n = len(poses)
    arr = (ctypes.c_int32 * (n * 7))()
    for i, (position, orientation) in enumerate(poses):
        x, y, z = position
        qw, qx, qy, qz = orientation
        base = 7 * i
        arr[base] = int(x); arr[base + 1] = int(y); arr[base + 2] = int(z)
        arr[base + 3] = int(qw); arr[base + 4] = int(qx); arr[base + 5] = int(qy); arr[base + 6] = int(qz)
    return arr, n


def compress_pose(poses) -> bytes:
    """6-DOF pose stream: position via the Geo3D auto-select, orientation
    via the Quaternion Joint (see DESIGN.md). poses: an iterable of
    ((x, y, z), (qw, qx, qy, qz)) pairs."""
    arr, n = _poses_to_array(poses)
    buf = _lib.csa_compress_pose(arr, n)
    return _check_and_extract(buf)


def compress_pose_lossy(poses, pos_quant_step: int, pos_resync_interval: int = 0,
                         quat_quant_step: int = 1, quat_resync_interval: int = 0) -> bytes:
    """pos_quant_step/pos_resync_interval reach the position half's existing
    lossy support; quat_quant_step/quat_resync_interval are the analogous
    knobs for the Quaternion Joint. Either quant_step <= 1 is lossless for
    that half."""
    arr, n = _poses_to_array(poses)
    buf = _lib.csa_compress_pose_lossy(arr, n, pos_quant_step, pos_resync_interval,
                                        quat_quant_step, quat_resync_interval)
    return _check_and_extract(buf)


def decompress_pose(data: bytes):
    """Returns a list of ((x, y, z), (qw, qx, qy, qz)) pairs. Works for both
    lossless and lossy blobs."""
    out_count = ctypes.c_size_t(0)
    buf = _lib.csa_decompress_pose(data, len(data), ctypes.byref(out_count))
    raw = _check_and_extract(buf)
    n = out_count.value
    if n == 0:
        return []
    values = struct.unpack(f"={7 * n}i", raw)
    return [((values[7 * i], values[7 * i + 1], values[7 * i + 2]),
             (values[7 * i + 3], values[7 * i + 4], values[7 * i + 5], values[7 * i + 6])) for i in range(n)]


def cuda_available() -> bool:
    return bool(_lib.csa_cuda_available())
