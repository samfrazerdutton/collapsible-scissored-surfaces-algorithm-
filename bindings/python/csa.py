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
from typing import Iterable, List, Tuple

__all__ = [
    "CsaError", "compress", "decompress",
    "compress_geo2d", "compress_geo2d_lossy", "decompress_geo2d",
    "compress_geo3d", "compress_geo3d_lossy", "decompress_geo3d",
    "compress_pose", "compress_pose_lossy", "decompress_pose",
    "cuda_available", "inspect", "verify",
]

# Kept in sync with pyproject.toml's [project].version by hand -- there is
# no build step that generates this file, so this is a real but manually
# maintained fact, not a guarantee the two can never drift for a commit or
# two. See DESIGN.md's "What this doc deliberately does not promise"
# section in FORMAT.md for the parallel caveat on wire-format stability.
__version__ = "0.1.0"

# Mode enum values, mirrored from include/csa/codec.hpp -- kept here (not
# imported) because there is no Python module wrapping that header, only
# this ctypes FFI layer.
_MODE_NAMES = {
    0: "Raw", 1: "General", 2: "Geo2D", 3: "Geo3D",
    4: "GeneralLZ", 5: "GeneralBWT", 6: "Pose",
}
_DIMS_SHAPES = {2: "geo2d", 3: "geo3d", 7: "pose"}


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
        # A real `pip install .` (scikit-build-core, see pyproject.toml)
        # places the compiled shared library right next to this module in
        # site-packages -- checked first since that's the common case for
        # anyone who didn't clone the repo and build it by hand.
        candidate = os.path.join(here, name)
        if os.path.exists(candidate):
            return candidate
        # Dev-repo fallback: running straight out of a git checkout with
        # cmake --build build done separately (the pattern every bench/
        # script and this project's own tests already use).
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


def _points2d_to_array(points: Iterable[Tuple[int, int]]):
    points = list(points)
    n = len(points)
    arr = (ctypes.c_int32 * (n * 2))()
    for i, (x, y) in enumerate(points):
        arr[2 * i] = int(x)
        arr[2 * i + 1] = int(y)
    return arr, n


def compress_geo2d(points: Iterable[Tuple[int, int]]) -> bytes:
    """points: an iterable of (x, y) integer-ish coordinate pairs."""
    arr, n = _points2d_to_array(points)
    buf = _lib.csa_compress_geo2d(arr, n)
    return _check_and_extract(buf)


def compress_geo2d_lossy(points: Iterable[Tuple[int, int]], quant_step: int, resync_interval: int = 0) -> bytes:
    """quant_step <= 1 is lossless. resync_interval periodically stores an
    exact rod to bound how far absolute position error can drift (0 =
    never); see DESIGN.md for the mechanism."""
    arr, n = _points2d_to_array(points)
    buf = _lib.csa_compress_geo2d_lossy(arr, n, quant_step, resync_interval)
    return _check_and_extract(buf)


def decompress_geo2d(data: bytes) -> List[Tuple[int, int]]:
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


def _points3d_to_array(points: Iterable[Tuple[int, int, int]]):
    points = list(points)
    n = len(points)
    arr = (ctypes.c_int32 * (n * 3))()
    for i, (x, y, z) in enumerate(points):
        arr[3 * i] = int(x)
        arr[3 * i + 1] = int(y)
        arr[3 * i + 2] = int(z)
    return arr, n


def compress_geo3d(points: Iterable[Tuple[int, int, int]]) -> bytes:
    """points: an iterable of (x, y, z) integer-ish coordinate triples."""
    arr, n = _points3d_to_array(points)
    buf = _lib.csa_compress_geo3d(arr, n)
    return _check_and_extract(buf)


def compress_geo3d_lossy(points: Iterable[Tuple[int, int, int]], quant_step: int, resync_interval: int = 0) -> bytes:
    """Same design as compress_geo2d_lossy. Tries both the xy+z composition
    and the true 3D similarity joint and keeps whichever encodes smaller.
    quant_step <= 1 is lossless (identical to compress_geo3d)."""
    arr, n = _points3d_to_array(points)
    buf = _lib.csa_compress_geo3d_lossy(arr, n, quant_step, resync_interval)
    return _check_and_extract(buf)


def decompress_geo3d(data: bytes) -> List[Tuple[int, int, int]]:
    out_count = ctypes.c_size_t(0)
    buf = _lib.csa_decompress_geo3d(data, len(data), ctypes.byref(out_count))
    raw = _check_and_extract(buf)
    n = out_count.value
    if n == 0:
        return []
    values = struct.unpack(f"={3 * n}i", raw)
    return [(values[3 * i], values[3 * i + 1], values[3 * i + 2]) for i in range(n)]


Pose = Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]


def _poses_to_array(poses: Iterable[Pose]):
    """poses: an iterable of ((x, y, z), (qw, qx, qy, qz)) pairs -- position
    and a unit-quaternion orientation, both integer-ish."""
    poses = list(poses)
    n = len(poses)
    arr = (ctypes.c_int32 * (n * 7))()
    for i, (position, orientation) in enumerate(poses):
        x, y, z = position
        qw, qx, qy, qz = orientation
        base = 7 * i
        arr[base] = int(x); arr[base + 1] = int(y); arr[base + 2] = int(z)
        arr[base + 3] = int(qw); arr[base + 4] = int(qx); arr[base + 5] = int(qy); arr[base + 6] = int(qz)
    return arr, n


def compress_pose(poses: Iterable[Pose]) -> bytes:
    """6-DOF pose stream: position via the Geo3D auto-select, orientation
    via the Quaternion Joint (see DESIGN.md). poses: an iterable of
    ((x, y, z), (qw, qx, qy, qz)) pairs."""
    arr, n = _poses_to_array(poses)
    buf = _lib.csa_compress_pose(arr, n)
    return _check_and_extract(buf)


def compress_pose_lossy(poses: Iterable[Pose], pos_quant_step: int, pos_resync_interval: int = 0,
                         quat_quant_step: int = 1, quat_resync_interval: int = 0) -> bytes:
    """pos_quant_step/pos_resync_interval reach the position half's existing
    lossy support; quat_quant_step/quat_resync_interval are the analogous
    knobs for the Quaternion Joint. Either quant_step <= 1 is lossless for
    that half."""
    arr, n = _poses_to_array(poses)
    buf = _lib.csa_compress_pose_lossy(arr, n, pos_quant_step, pos_resync_interval,
                                        quat_quant_step, quat_resync_interval)
    return _check_and_extract(buf)


def decompress_pose(data: bytes) -> List[Pose]:
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


def _read_u64le(data: bytes, pos: int) -> Tuple[int, int]:
    if pos + 8 > len(data):
        raise CsaError("truncated .csa file: expected 8 more bytes for a u64 field")
    return struct.unpack_from("<Q", data, pos)[0], pos + 8


def inspect(data: bytes) -> dict:
    """Parses the real .csa wire-format headers (see FORMAT.md) without
    decoding the payload -- a pure-Python mirror of `scissorc inspect` for
    an actual .csa file (not a raw/undetected input; this module has no
    Python port of squeeze's text-table sniffing heuristic, so it only
    inspects bytes that already carry a CSAG or CSA1 magic).

    Returns a dict containing only fields that actually exist in the
    format:
      - always: "layer" (1 or 2), "mode" (int), "mode_name" (str),
        "payload_bytes" (int, the mode-specific payload this function does
        not further decode -- FORMAT.md doesn't specify those byte layouts
        either)
      - layer 2 only: "dims" (2/3/7), "shape" ("geo2d"/"geo3d"/"pose"),
        "scale" (int), and "qscale" (int, only when dims == 7)

    Raises CsaError if `data` doesn't start with either magic.
    """
    if len(data) >= 5 and data[0:4] == b"CSAG":
        dims = data[4]
        if dims not in _DIMS_SHAPES:
            raise CsaError(f"unrecognized CSAG dims byte: {dims}")
        pos = 5
        scale, pos = _read_u64le(data, pos)
        result = {"layer": 2, "dims": dims, "shape": _DIMS_SHAPES[dims], "scale": scale}
        if dims == 7:
            qscale, pos = _read_u64le(data, pos)
            result["qscale"] = qscale
        inner = data[pos:]
        if len(inner) < 5 or inner[0:4] != b"CSA1":
            raise CsaError("CSAG header present but no CSA1 magic follows it -- file may be truncated or corrupted")
        mode = inner[4]
        result.update({
            "mode": mode,
            "mode_name": _MODE_NAMES.get(mode, "unknown"),
            "payload_bytes": len(inner) - 5,
        })
        return result

    if len(data) >= 5 and data[0:4] == b"CSA1":
        mode = data[4]
        return {
            "layer": 1,
            "mode": mode,
            "mode_name": _MODE_NAMES.get(mode, "unknown"),
            "payload_bytes": len(data) - 5,
        }

    raise CsaError("not a recognized .csa file (no CSAG or CSA1 magic)")


def verify(data: bytes) -> dict:
    """Structural verification of a .csa blob on its own, without the
    original input to diff against -- the same thing `scissorc verify`
    checks. Attempts the real decode and returns {"shape": str, "count":
    int} on success. Raises CsaError with the underlying decoder's message
    on failure.

    This is NOT a claim that the result matches some original file
    bit-for-bit -- that check already happens inside squeeze() itself and
    needs the original to compare against, which a standalone .csa doesn't
    carry. What this can honestly promise: the decoder didn't detect
    internal inconsistency while decoding it (see FORMAT.md's "known gap:
    no CRC" -- this is not a checksum, just successful decode).
    """
    info = inspect(data)
    if info["layer"] == 2:
        dims = info["dims"]
        pos = 5 + 8 + (8 if dims == 7 else 0)
        blob = data[pos:]
        if dims == 2:
            pts = decompress_geo2d(blob)
            return {"shape": "geo2d", "count": len(pts)}
        elif dims == 3:
            pts = decompress_geo3d(blob)
            return {"shape": "geo3d", "count": len(pts)}
        else:
            poses = decompress_pose(blob)
            return {"shape": "pose", "count": len(poses)}
    else:
        restored = decompress(data)
        return {"shape": "general", "count": len(restored)}
