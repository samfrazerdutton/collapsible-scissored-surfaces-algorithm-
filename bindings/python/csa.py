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
import re
import struct
from typing import Iterable, List, Tuple

__all__ = [
    "CsaError", "compress", "decompress",
    "compress_geo2d", "compress_geo2d_lossy", "decompress_geo2d",
    "compress_geo3d", "compress_geo3d_lossy", "decompress_geo3d",
    "compress_pose", "compress_pose_lossy", "decompress_pose",
    "cuda_available", "inspect", "verify",
    "optimize_geo2d", "optimize_geo3d", "optimize_pose", "profile",
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


def _search_max_quant_step(budget: float, hi: int, eval_fn) -> int:
    """Largest quant_step in [1, hi] whose real measured error (from eval_fn,
    which must actually compress+decompress+measure, not estimate) stays
    <= budget. eval_fn is assumed monotonically non-decreasing in
    quant_step -- true of this codec's quantization by construction.
    Returns 0 if even quant_step=1 (the finest granularity the *lossy*
    path offers) already exceeds the budget, in which case the caller
    should recommend true lossless rather than a fabricated best effort.
    Mirrors cli/main.cpp's search_max_quant_step exactly."""
    if eval_fn(1) > budget:
        return 0
    lo, hi_, best = 1, hi, 1
    while lo <= hi_:
        mid = lo + (hi_ - lo) // 2
        if eval_fn(mid) <= budget:
            best = mid
            if mid == hi_:
                break
            lo = mid + 1
        else:
            if mid == lo:
                break
            hi_ = mid - 1
    return best


_OPTIMIZE_SEARCH_HI = 1 << 20


def optimize_geo2d(points: Iterable[Tuple[int, int]], max_pos_error: float) -> dict:
    """Auto-Optimize for geo2d: searches the real quant_step space (via
    actual compress+decompress+measure at each candidate -- see
    _search_max_quant_step) for the strongest compression whose measured
    max coordinate error stays within max_pos_error, in the same integer
    units `points` is already expressed in. Returns {"blob": bytes,
    "quant_step": int or None, "measured_error": float, "lossless": bool}
    -- lossless=True means even the finest lossy step exceeded the
    budget and true compress_geo2d() was used instead."""
    points = list(points)

    def eval_fn(step):
        blob = compress_geo2d_lossy(points, step, 64)
        back = decompress_geo2d(blob)
        return max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(points, back)), default=0.0)

    step = _search_max_quant_step(max_pos_error, _OPTIMIZE_SEARCH_HI, eval_fn)
    if step == 0:
        return {"blob": compress_geo2d(points), "quant_step": None, "measured_error": 0.0, "lossless": True}
    blob = compress_geo2d_lossy(points, step, 64)
    return {"blob": blob, "quant_step": step, "measured_error": eval_fn(step), "lossless": False}


def optimize_geo3d(points: Iterable[Tuple[int, int, int]], max_pos_error: float) -> dict:
    """Auto-Optimize for geo3d. See optimize_geo2d -- identical contract."""
    points = list(points)

    def eval_fn(step):
        blob = compress_geo3d_lossy(points, step, 64)
        back = decompress_geo3d(blob)
        return max((max(abs(a - b) for a, b in zip(p, q)) for p, q in zip(points, back)), default=0.0)

    step = _search_max_quant_step(max_pos_error, _OPTIMIZE_SEARCH_HI, eval_fn)
    if step == 0:
        return {"blob": compress_geo3d(points), "quant_step": None, "measured_error": 0.0, "lossless": True}
    blob = compress_geo3d_lossy(points, step, 64)
    return {"blob": blob, "quant_step": step, "measured_error": eval_fn(step), "lossless": False}


def optimize_pose(poses: Iterable[Pose], max_pos_error: float = None, max_quat_error: float = None) -> dict:
    """Auto-Optimize for 6-DOF pose: position and rotation are searched
    independently (holding the other at quant_step=1, since the format
    encodes them as two genuinely separate sub-streams -- see FORMAT.md's
    Pose mode description), then the combined configuration is
    re-verified for real before being returned, rather than assumed
    additive. Pass only one of max_pos_error/max_quat_error to optimize
    just that half (the other is held at quant_step=1, the finest the
    lossy path offers). Mirrors `scissorc optimize` exactly.

    Returns {"blob": bytes, "pos_quant_step": int or None,
    "quat_quant_step": int or None, "measured_pos_error": float,
    "measured_quat_error": float, "pos_lossless": bool, "quat_lossless":
    bool}."""
    if max_pos_error is None and max_quat_error is None:
        raise ValueError("optimize_pose needs at least one of max_pos_error / max_quat_error")
    poses = list(poses)

    def eval_pos(step):
        blob = compress_pose_lossy(poses, step, 64, 1, 64)
        back = decompress_pose(blob)
        return max((max(abs(a - b) for a, b in zip(p[0], q[0])) for p, q in zip(poses, back)), default=0.0)

    def eval_quat(step):
        blob = compress_pose_lossy(poses, 1, 64, step, 64)
        back = decompress_pose(blob)
        return max((max(abs(a - b) for a, b in zip(p[1], q[1])) for p, q in zip(poses, back)), default=0.0)

    pos_step = _search_max_quant_step(max_pos_error, _OPTIMIZE_SEARCH_HI, eval_pos) if max_pos_error is not None else 1
    quat_step = _search_max_quant_step(max_quat_error, _OPTIMIZE_SEARCH_HI, eval_quat) if max_quat_error is not None else 1
    pos_lossless = max_pos_error is not None and pos_step == 0
    quat_lossless = max_quat_error is not None and quat_step == 0

    if pos_lossless and quat_lossless:
        return {
            "blob": compress_pose(poses), "pos_quant_step": None, "quat_quant_step": None,
            "measured_pos_error": 0.0, "measured_quat_error": 0.0, "pos_lossless": True, "quat_lossless": True,
        }

    final_pos_step = 1 if pos_step == 0 else pos_step
    final_quat_step = 1 if quat_step == 0 else quat_step
    blob = compress_pose_lossy(poses, final_pos_step, 64, final_quat_step, 64)
    back = decompress_pose(blob)
    combined_pos_err = max((max(abs(a - b) for a, b in zip(p[0], q[0])) for p, q in zip(poses, back)), default=0.0)
    combined_quat_err = max((max(abs(a - b) for a, b in zip(p[1], q[1])) for p, q in zip(poses, back)), default=0.0)
    if (max_pos_error is not None and combined_pos_err > max_pos_error) or \
       (max_quat_error is not None and combined_quat_err > max_quat_error):
        raise CsaError(
            f"internal error: combined configuration violated a budget that passed in isolation "
            f"(pos_err={combined_pos_err}, quat_err={combined_quat_err}) -- please report this"
        )
    return {
        "blob": blob,
        "pos_quant_step": None if pos_lossless else final_pos_step,
        "quat_quant_step": None if quat_lossless else final_quat_step,
        "measured_pos_error": combined_pos_err, "measured_quat_error": combined_quat_err,
        "pos_lossless": pos_lossless, "quat_lossless": quat_lossless,
    }


def _find_cli() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    name = "scissorc.exe" if platform.system() == "Windows" else "scissorc"
    for candidate in (
        os.path.join(here, name),
        os.path.abspath(os.path.join(here, "..", "..", "build", name)),
    ):
        if os.path.exists(candidate):
            return candidate
    raise OSError(
        "Could not locate the scissorc CLI binary (checked next to this file and "
        "../../build/ relative to it). profile() shells out to it rather than "
        "reimplementing its text-table sniffing heuristic a second time in Python -- "
        "build the project first (cmake --build build --target scissorc)."
    )


def profile(path: str) -> dict:
    """Runs `scissorc inspect <path>` and parses its real output into a
    structured dict. Deliberately implemented by shelling out to the
    actual CLI rather than re-implementing its text-table detection
    heuristic (sniff_table(), cli/main.cpp) a second time in pure Python
    -- that heuristic lives in exactly one place, already covered by the
    C++ test suite, so this can't drift out of sync with what
    `scissorc squeeze` itself would actually detect. Requires the CLI
    binary to be built and locatable (see _find_cli()) -- a real
    limitation of this specific function, not the rest of the module: a
    plain `pip install .` only packages libcsa itself (see
    pyproject.toml's build.targets), not the scissorc executable, so
    profile() raises OSError with an actionable message in that case
    while every other function in this module keeps working normally.
    This is a different code path than the rest of this module, which
    talks to libcsa directly via ctypes and has no such dependency.

    Returns a dict with at least "recognized_csa_file" (bool). For a raw
    (non-.csa) input: "shape" ("general"/"geo2d"/"geo3d"/"pose"),
    "confidence_pct" (float), "recommended" (bool), plus
    "decimals_position"/"decimals_orientation"/"max_abs_position"/
    "max_abs_orientation" when a shape was detected. For an actual .csa
    file, returns the same fields inspect() already provides.
    """
    import subprocess

    cli = _find_cli()
    result = subprocess.run([cli, "inspect", path], capture_output=True, text=True)
    output = result.stdout
    if result.returncode != 0:
        raise CsaError(f"scissorc inspect failed: {result.stderr.strip() or output.strip()}")

    if "not a recognized .csa file" not in output:
        # Real .csa file -- read it and reuse inspect() rather than re-parsing this text a second way.
        with open(path, "rb") as f:
            info = inspect(f.read())
        info["recognized_csa_file"] = True
        return info

    info = {"recognized_csa_file": False}
    m = re.search(r"detected shape: general", output)
    if m:
        info["shape"] = "general"
        conf_m = re.search(r"match rate: (\d+) / (\d+) lines \(([\d.]+)%", output)
        if conf_m:
            info["confidence_pct"] = float(conf_m.group(3))
        info["recommended"] = False
        return info

    shape_m = re.search(r"detected shape: (geo2d|geo3d|pose)", output)
    if shape_m:
        info["shape"] = shape_m.group(1)
        conf_m = re.search(r"confidence: (\d+) / (\d+) lines matched \(([\d.]+)%\)", output)
        if conf_m:
            info["confidence_pct"] = float(conf_m.group(3))
        dec_m = re.search(r"decimal precision: position=(\d+)(?:, orientation=(\d+))?", output)
        if dec_m:
            info["decimals_position"] = int(dec_m.group(1))
            if dec_m.group(2):
                info["decimals_orientation"] = int(dec_m.group(2))
        abs_m = re.search(r"max abs value: position=([\d.]+)(?:, orientation=([\d.]+))?", output)
        if abs_m:
            info["max_abs_position"] = float(abs_m.group(1))
            if abs_m.group(2):
                info["max_abs_orientation"] = float(abs_m.group(2))
        info["recommended"] = True
        return info

    info["raw_output"] = output
    return info
