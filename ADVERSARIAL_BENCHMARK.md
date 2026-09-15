# CSA vs. the tools a robotics/AV team actually reaches for today

`REAL_POSE_BENCHMARK.md` and `REAL_GEO_BENCHMARK.md` benchmark CSA against
general-purpose compressors (gzip/bz2/lzma/zstd/brotli) and, on the LiDAR
side, the actual specialized incumbent (LASzip). Neither compares against
what a real robotics/AV/teleoperation team logging 6-DOF pose or point-cloud
data today would actually put in their pipeline: **MCAP** (Foxglove's
container format, the rosbag2 default) with its default **zstd** chunk
compression for telemetry, and **Draco** (Google's geometry codec) for point
clouds. This file adds those two, on the exact same real datasets, generated
by `bench/adversarial_benchmark.py`. Losses are reported here exactly as
honestly as wins -- see `REAL_GEO_BENCHMARK.md` for the one real loss this
project already documents (LASzip beats CSA decisively on raw LiDAR
geometry); this file's job is to check to two more real competitors, not to
find only the comparisons that look good.

## Result 1: 6-DOF pose streams vs. MCAP+zstd (real ROS2 messages)

Each pose was written as a real `geometry_msgs/msg/PoseStamped` ROS2
message -- the actual `.msg` schema, full float64 precision, a real
per-message timestamp and `frame_id: "map"` -- into an MCAP file using its
default zstd chunk compression (`chunk_size` set generously large so every
dataset here fits in a single zstd chunk, the fairest possible case for it,
not a chunking artifact working against it). This is not a stripped-down
number: it's what a team recording this topic via rosbag2/Foxglove actually
gets on disk.

| dataset | raw packed (7×int32) | MCAP+zstd (real PoseStamped) | CSA `compress-pose` (lossless) | CSA vs. MCAP+zstd |
|---|---:|---:|---:|---:|
| EuRoC V1_02 (16,702 real poses, drone flight) | 467,656 | 1,208,893 (**158.5% larger** than raw) | 146,999 (68.6% smaller than raw) | **CSA 87.8% smaller** |
| TUM fr2/desk (20,957 real poses, handheld camera) | 586,796 | 960,637 (**63.7% larger** than raw) | 212,162 (63.8% smaller than raw) | **CSA 77.9% smaller** |
| KITTI 00 (4,541 real poses, vehicle driving) | 127,148 | 355,270 (**179.4% larger** than raw) | 72,093 (43.3% smaller than raw) | **CSA 79.7% smaller** |

### Why MCAP+zstd loses this badly, honestly

This is not "zstd is a bad compressor." zstd is doing its job; the problem
is upstream of it. A `PoseStamped` message's CDR-serialized payload is
already ~2.5-3x larger than CSA's own raw-packed int32 baseline *before any
compression happens*, for two structural reasons: (1) every position/
orientation component is a float64 (8 bytes) where CSA's baseline (and
internal representation) uses a 4-byte int32 at the precision the source
data actually has; (2) every message repeats a full header -- a `sec`/
`nanosec` timestamp and a `frame_id` string -- that a continuous pose stream
doesn't need per-sample at all. zstd's chunk-level compression finds real
redundancy *within* that representation (repeated `frame_id` bytes, similar
neighboring float patterns), which is exactly why the result isn't even
worse -- but general-purpose entropy coding over a byte stream has no way to
know that consecutive quaternions differ by a small, smoothly-varying
delta rotation, or that position deltas between samples are far smaller
than the positions themselves. That's precisely the structure CSA's
Quaternion Joint and Rod-Joint transforms are built to exploit, and it is
invisible to a general-purpose byte-level compressor no matter how good it
is. The honest framing: **the message format's fixed representational
overhead, not zstd, is the dominant cost here for small pose messages** --
and no amount of swapping the compressor underneath that format fixes it.

## Result 2: LiDAR point cloud vs. Draco (real airborne scan, 693,895 points)

Draco (`DracoPy`, `compression_level=10`) is lossy by design -- there is no
"lossless Draco" setting -- so three quantization levels are reported
rather than one cherry-picked number. Quantization *step* below is the
guaranteed precision derived from the real bounding box and `quantization_bits`
(not a measured error: Draco's decoder reorders/deduplicates points relative
to the input, so an index-wise error comparison isn't valid; the analytic
step size is the rigorous, decoder-order-independent way to state it).
Coordinates were centered (mean subtracted) before encoding -- see below
for why that step turned out to matter far more than expected.

| method | size (bytes) | vs. raw packed | precision | points preserved | round-trip |
|---|---:|---:|---:|---:|:---:|
| raw packed (4-byte int × 3 axes) | 8,326,740 | -- | exact | 693,895 / 693,895 | (reference) |
| Draco (qbits=11) | 562,968 | 93.2% smaller | step ≈170.0mm | 693,875 / 693,895 | lossy, 20 points dropped |
| Draco (qbits=16) | 1,833,481 | 78.0% smaller | step ≈5.3mm | 693,875 / 693,895 | lossy, 20 points dropped |
| Draco (qbits=20) | 2,853,172 | 65.7% smaller | step ≈0.33mm | 693,875 / 693,895 | lossy, 20 points dropped |
| **CSA `compress-geo3d`** | **1,336,093** | **84.0% smaller** | **exact** | **693,895 / 693,895** | **PASS, exact** |
| **LASzip (geometry-only)** | **1,012,384** | **87.8% smaller** | **exact** | **693,895 / 693,895** | **PASS, exact** |

### A real bug in this benchmark, found and fixed -- and the actual finding underneath it

The first version of this table fed Draco the scan's raw absolute UTM
coordinates directly (as most naive point-cloud pipelines would) and
found something alarming: **2,264 of 693,895 points dropped on decode, at
every quantization level tested, including the finest (qbits=20, a
guaranteed ~0.33mm step)**. That number was wrong, and here's the real
mechanism, tracked down rather than left as "huh, weird":

DracoPy's point-cloud encoder takes float32/float64 NumPy input. This
scan's real coordinates are large absolute UTM values (~4.9x10^5 m on the
X axis). **float32 has ~7 significant decimal digits of precision** -- at
a magnitude of 500,000, its own representable precision is only about
**500,000 x 2^-23 ~ 6cm**, which is *coarser than the source data's real
1cm stored precision*, and that precision is lost the moment the
coordinates are cast to float32 -- **before `quantization_bits` ever gets
a chance to matter**. No amount of raising `quantization_bits` can recover
precision the input array had already lost. Centering the coordinates
first (subtracting the mean, so values are O(100m) instead of O(5x10^5m))
gives float32 enough headroom to represent 1cm precision cleanly, and the
dropped-point count falls to exactly 20 -- which matches, precisely, the
20 real exact-duplicate XYZ triples already present in the raw source
data (checked directly against the LAS file's own integer coordinates).
Confirmed by varying only that one thing (centered vs. not) with
`quantization_bits` and `compression_level` both held fixed: the drop
count tracks the centering, not the quantization setting.

**This is not really a Draco-specific bug -- it's a real, common, easy-to-
miss integration pitfall for *any* float32-based geometry pipeline fed
absolute UTM/ECEF-style coordinates**, and it is structurally exactly the
problem LAS/LAZ's own file format design (a per-file float64
scale+offset pair plus int32 stored coordinates) exists to sidestep: never
hand a lossy, magnitude-dependent-precision float32 value a huge absolute
offset it doesn't need to carry per-point. CSA never hits this failure
mode for the same structural reason -- its `--scale`-quantized
representation is exact fixed-point integers throughout, and squeeze's
auto-detected scale (`cli/main.cpp`'s `safe_scale`) is chosen from the
data's own precision, not from an arbitrary float32 cast. The corrected,
properly-centered comparison above is the fair one; the caught version of
this bug is itself the more interesting engineering lesson.

### Combined honest picture

- With Draco integrated correctly (centered coordinates), its remaining
  "loss" (20 points) is arguably not a loss at all -- those are the
  dataset's own genuine exact duplicates, which CSA and LASzip also collapse
  to identical output values (just without ever dropping the row).
- CSA beats Draco on ratio at Draco's higher-precision settings (16/20-bit:
  CSA's 1,336,093 bytes beats both, while also being the only result of
  the three with zero rows dropped). At Draco's roughest setting (11-bit,
  ~170mm precision -- coarser than useful for most real LiDAR applications),
  Draco is smaller than CSA.
- LASzip still wins outright on this dataset, exactly as `REAL_GEO_BENCHMARK.md`
  already found -- smaller than CSA *and* exact *and* ~75x faster to
  compress. That finding is unchanged by adding Draco to the table; Draco is
  a mesh/point-cloud geometry codec, not built for airborne-scan-order
  structure the way LASzip is, and lands between CSA and general-purpose
  compressors on ratio.

## Regenerating these numbers

```
pip install mcap mcap-ros2-support DracoPy
python bench/adversarial_benchmark.py
```

Needs the same real datasets `REAL_POSE_BENCHMARK.md`/`REAL_GEO_BENCHMARK.md`
already document how to fetch (EuRoC/TUM/KITTI ground truth under
`bench/_thirdparty/pose_data/`, `autzen/stadium-utm.laz` under
`bench/_thirdparty/lidar_data/`) plus a built `scissorc` (`cmake --build build`).
