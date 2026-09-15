# The `.csa` format

**Status: pre-1.0. The format may change without notice, without a
migration path, and without preserving compatibility with files written
by an earlier version of this codebase.** There is no version negotiation
in the format today -- a reader can detect the *mode* of a blob (see
below), but there is no field that says "this blob was written by v0.3.1,
here's how to handle v0.2.0 blobs differently." Until there is (a real,
versioned stability promise is v1.0's actual job, not a documentation
exercise), don't build anything that needs old `.csa` files to keep
decoding after an upgrade. This doc describes the format exactly as it
exists in the code today, including the rough edges, not an idealized
target.

All integers are little-endian. All of this is derived directly from
`include/csa/codec.hpp`, `src/codec.cpp`, `cli/main.cpp`, and
`src/rans_coder.cpp` -- if this doc and the code ever disagree, the code
is right and this doc has a bug; please file it.

## Layer 1: the library-level container (`"CSA1"` + mode byte)

Every blob produced by `compress()`, `compress_geo2d()`, `compress_geo3d()`,
`compress_pose()` (and their lossy variants) -- the actual C++ library/C
ABI functions, `codec.cpp`'s `write_magic_mode`/`read_magic_mode` -- starts
with:

| offset | size | field |
|---|---|---|
| 0 | 4 bytes | magic: `"CSA1"` (`0x43 0x53 0x41 0x31`) |
| 4 | 1 byte | `Mode` (see below) |
| 5+ | varies | mode-specific payload |

`Mode` (`include/csa/codec.hpp`):

| value | name | produced by | payload |
|---|---|---|---|
| 0 | `Raw` | `compress()`, when nothing else won | `u64` length + the input bytes, verbatim |
| 1 | `General` | `compress()`, when the Pantograph Lift won | a serialized `LiftResult` (see `pantograph_lift.hpp`) |
| 2 | `Geo2D` | `compress_geo2d()`/`compress_geo2d_lossy()` | a serialized `RodJoint2DResult` (quant_step/resync_interval ride inside it -- a lossy blob decodes with the same `decompress_geo2d()` as a lossless one, no separate lossy decode path) |
| 3 | `Geo3D` | `compress_geo3d()`/`compress_geo3d_lossy()` | one sub-mode byte (`0`=xy-rotation+z-affine composition, `1`=true 3D similarity joint) then that model's serialized result |
| 4 | `GeneralLZ` | `compress()`, when the LZ dictionary matcher won | LZ token stream + entropy-coded payload (`lz_codec.hpp`) |
| 5 | `GeneralBWT` | `compress()`, when BWT+MTF won (inputs up to `kBwtMaxInputSize`) | BWT-transformed, MTF+RUNA/RUNB-encoded payload (`bwt_codec.hpp`) |
| 6 | `Pose` | `compress_pose()`/`compress_pose_lossy()` | a serialized Geo3D position result followed by a serialized Quaternion Joint orientation result |

`decompress()` reads the mode byte and dispatches; `decompress_geo2d()`/
`decompress_geo3d()`/`decompress_pose()` each check the mode byte matches
what they expect and throw `std::runtime_error` otherwise (`"csa: not a
Geo2D stream"` etc.) -- there's no silent misinterpretation of the wrong
blob type.

**Known gap**: there's no CRC or checksum anywhere in this layer. A
truncated or corrupted blob is only caught if it happens to make an
internal length field inconsistent enough to throw; it is not guaranteed
to be caught. Don't rely on this format to detect its own corruption.

## Layer 2: the CLI/browser file convention (`"CSAG"` + dims + scale)

This layer is **not** part of the C ABI or the core library -- it's a
convention `cli/main.cpp` (the `compress-geo2d`/`compress-geo3d`/
`compress-pose` commands and `squeeze`/`unsqueeze`) and `docs/index.html`'s
JS port of the same logic both implement, because the library itself only
ever sees pre-quantized integers -- it has no idea a `Geo2D` blob's
integers were originally meters scaled by 1,000,000 versus centimeters
scaled by 100. Something has to remember that, and this is that something:

| offset | size | field |
|---|---|---|
| 0 | 4 bytes | magic: `"CSAG"` (`0x43 0x53 0x41 0x47`) |
| 4 | 1 byte | `dims`: `2` (geo2d), `3` (geo3d), or `7` (pose) |
| 5 | 8 bytes | `scale` (`u64`, little-endian) -- position fixed-point scale |
| 13 | 8 bytes | `qscale` (`u64`, little-endian) -- **only present when `dims == 7`** -- orientation fixed-point scale |
| 13 (or 21 if `dims==7`) | remainder | the Layer 1 blob, unmodified -- i.e. this literally contains a second `"CSA1"` magic + mode byte inside it |

So a real `.csa` file produced by `scissorc squeeze somefile.csv` on a
detected 3-column (geo3d) input looks like:
`"CSAG" | 0x03 | scale (8 bytes) | "CSA1" | 0x03 (Geo3D) | 0x00-or-0x01 (sub-mode) | ...`
-- two magic numbers back to back. This is a real, slightly redundant
artifact of Layer 2 having been added on top of an existing Layer 1 that
already had its own framing, not a deliberately designed nesting. It's
documented here as-is because that's what's actually on disk, not because
it's the cleanest possible design.

General (non-numeric) files handled by `squeeze` have **no Layer 2 at
all** -- they're a bare Layer 1 blob (`"CSA1"` + `Mode::General`/`GeneralLZ`/
`GeneralBWT`/`Raw`), since there's no scale/qscale to remember for
arbitrary bytes. `unsqueeze` (and the browser port) detect which layer
they're looking at by checking for `"CSAG"` first, falling back to
treating the file as a bare Layer 1 blob otherwise.

## The interleaved rANS format (standalone, no relation to Layers 1/2)

`encode_interleaved_rans()`/`decode_interleaved_rans()` (`rans_coder.hpp`,
`csa_rans_encode`/`csa_rans_decode` in the C ABI) produce a **completely
separate** wire format with no relation to `"CSA1"`/`"CSAG"`:

| offset | size | field |
|---|---|---|
| 0 | 8 bytes | `total_len` (`u64`) -- decoded output length |
| 8 | 4 bytes | `num_lanes` (`u32`) |
| 12 | 4 bytes | `scale_bits` (`u32`) |
| 16 | 256 varints | quantized frequency table (one LEB128-style varint per byte value 0-255) |
| ... | per lane, repeated `num_lanes` times | `u64 symcount`, `u64 bytelen`, then `bytelen` bytes of that lane's rANS-coded stream |

**Known gap, worth fixing before this format matures**: this has **no
magic number at all**. There is currently no way to look at an arbitrary
byte blob and tell "this is an interleaved-rANS blob" versus "this is a
Layer 1 or Layer 2 CSA blob" versus "this is neither" -- the caller has to
already know which function produced it. If this format is going to be
used anywhere a reader might encounter it without out-of-band context
(a real integration point, not just this repo's own tests calling its own
encoder), it needs its own magic prefix first.

## What this doc deliberately does not promise

- **No backward compatibility across versions yet.** Every mode's
  internal payload format (the `LiftResult`/`RodJoint2DResult`/etc.
  serializations this doc doesn't expand into byte-level detail) is
  subject to change as the transforms themselves change. This doc
  describes the container framing, which is the part a third-party reader
  would actually need to parse to dispatch to the right decoder -- not a
  byte-level spec of every internal transform's serialization, which
  changes with the code and is the actual implementation, not a format.
- **No forward compatibility.** An older reader given a blob from a newer
  version has no guaranteed way to detect that gracefully today beyond
  "an unrecognized mode byte throws."
- **No integrity guarantee**, per the Layer 1 note above.

A real 1.0 stability promise means fixing these three things first, not
just declaring a version number.
