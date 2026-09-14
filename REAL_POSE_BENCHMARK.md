# CSA's 6-DOF pose codec vs. real tracking data

`REAL_GEO_BENCHMARK.md` found the Rod-Joint Transform's calibrated-rotation
model fits real *trajectory*-shaped data (a GPS trip) well but real
*point-cloud*-shaped data (a LiDAR scan) poorly. That finding motivated
building a Quaternion Joint transform for the *orientation* half of a 6-DOF
pose stream (see `DESIGN.md`), on the theory that real tracking data --
VR/AR headsets, drones, robots, SLAM cameras -- has locally consistent
angular velocity the same way real vehicle/GPS paths have locally
consistent direction. Until now that codec (`compress_pose`) had only been
measured on synthetic pose data. This file is the real-data check.

There is no dominant specialized "6-DOF pose codec" the way LASzip is for
LiDAR or Google's Encoded Polyline format is for 2D GPS traces -- 6-DOF
tracking data is usually stored as plain CSV/text (as the three real
datasets below are, straight from their original sources) and compressed
generically if at all. That absence is itself part of why this is a
less-contested niche; general-purpose compressors (gzip/bz2/lzma) are the
only fair reference available, so that's what this file compares against.

## Data provenance (read this before trusting the numbers)

Three real trajectories, three different kinds of real motion, all
ground-truth pose logs (not synthetic, not generated for this project),
mirrored as small standalone files by the
[`evo`](https://github.com/MichaelGrupp/evo) trajectory-evaluation
library's test suite -- `evo` sources them from the original dataset
releases; using its mirror avoids the multi-hundred-MB-to-multi-GB
image/video bundles the original EuRoC/TUM/KITTI downloads ship the
ground truth *inside*, which would otherwise make this benchmark
impractical to regenerate.

- **EuRoC MAV, Vicon Room 1 "02"** (real drone flight, motion-capture
  ground truth) -- `V102_groundtruth.csv`, the *original* EuRoC CSV at
  ~200Hz IMU rate, 16,702 real poses. Columns are already
  `(t, p_x,p_y,p_z, q_w,q_x,q_y,q_z)` -- no quaternion reordering needed.
- **TUM RGB-D, freiburg2/desk** (a real handheld camera moved by hand
  around a desk, motion-capture ground truth) -- `fr2_desk_groundtruth.txt`,
  20,957 real poses, TUM format `t tx ty tz qx qy qz qw` -- quaternion
  reordered from (x,y,z,w) to (w,x,y,z) below.
- **KITTI odometry, sequence 00** (a real car driving through Karlsruhe,
  RTK-GPS/IMU-derived ground truth) -- `KITTI_00_gt.txt`, 4,541 real poses,
  stored as flattened 3x4 rotation+translation matrices, converted to
  quaternions via a standard branch-based (Shepperd's method) routine --
  verified during development by reconstructing the rotation matrix from
  each converted quaternion and comparing to the original: max entrywise
  error ~2.5e-7 across all 4,541 real rotations.

All three fetched directly from `evo`'s GitHub raw file host (URLs in
`bench/real_pose_benchmark.py`'s module docstring). Every quaternion's
unit-norm was sanity-checked after conversion (all within ~1.5e-4 of norm
1, consistent with real noisy motion-capture/RTK data, not a conversion
bug), and the KITTI conversion was independently verified as above before
any benchmark number was trusted.

## Results

`SCALE = QSCALE = 1,000,000` throughout (preserves each source file's own
quoted ~6-decimal-digit precision exactly -- lossless at that precision).
"raw packed" is 7 int32s per pose (x,y,z,qw,qx,qy,qz), the fair baseline
BENCHMARKS.md/REAL_GEO_BENCHMARK.md already establish for this kind of
comparison. Every CSA number below is round-trip verified (compress,
decompress, compare) before being trusted; `compress-pose-lossy` uses one
representative setting (`--pos-quant 8 --pos-resync 64 --quat-quant 32
--quat-resync 64`) across all three, not tuned per-dataset.

### EuRoC V1_02 (16,702 real poses, drone flight)

| method | size (bytes) | vs. raw packed |
|---|---:|---:|
| raw packed | 467,656 | -- |
| gzip -9 | 365,071 | 21.9% smaller |
| bz2 -9 | 350,970 | 25.0% smaller |
| lzma -9 | 230,840 | 50.6% smaller |
| **CSA `compress-pose`** | **160,151** | **65.8% smaller** |
| CSA `compress-pose-lossy` | 65,504 | 86.0% smaller (max pos err 9.4e-5, max orientation-component err 1.6e-5) |

Position-only (`compress-geo3d` on the same points): 63,399 bytes.
Orientation's share of the full pose blob: 96,752 bytes.

### TUM fr2/desk (20,957 real poses, handheld camera)

| method | size (bytes) | vs. raw packed |
|---|---:|---:|
| raw packed | 586,796 | -- |
| gzip -9 | 324,979 | 44.6% smaller |
| bz2 -9 | 309,731 | 47.2% smaller |
| lzma -9 | 223,080 | 62.0% smaller |
| **CSA `compress-pose`** | **212,295** | **63.8% smaller** |
| CSA `compress-pose-lossy` | 135,322 | 76.9% smaller (max pos err 7.4e-5, max orientation-component err 1.6e-5) |

Position-only: 90,271 bytes. Orientation's share: 122,024 bytes.

### KITTI odometry 00 (4,541 real poses, vehicle driving)

| method | size (bytes) | vs. raw packed |
|---|---:|---:|
| raw packed | 127,148 | -- |
| gzip -9 | 107,288 | 15.6% smaller |
| bz2 -9 | 109,382 | 14.0% smaller |
| lzma -9 | 78,332 | 38.4% smaller |
| **CSA `compress-pose`** | **74,590** | **41.3% smaller** |
| CSA `compress-pose-lossy` | 51,618 | 59.4% smaller (max pos err 7.7e-5, max orientation-component err 1.6e-5) |

Position-only: 36,264 bytes. Orientation's share: 38,326 bytes.

## Honest verdict

- **CSA beats lzma -9 -- the strongest general-purpose reference tested
  -- on all three real trajectories**, not just the one it was designed
  around: 30.6% smaller on the drone flight, 4.8% smaller on the
  handheld camera, 4.8% smaller on the vehicle drive. This is a real,
  consistent (if not equally large) win across three genuinely different
  kinds of real 6-DOF motion, not a single cherry-picked case -- the
  closest this project has come to validating the "less-contested niche"
  bet plainly.

- **The win is largest on the smoothest motion and smallest on the least
  smooth**, exactly as the model predicts. The EuRoC drone flight (a
  motion-capture-tracked MAV under stabilized flight control) is the
  smoothest of the three and gets the biggest win (30.6% over lzma). The
  handheld camera and vehicle drive both involve more irregular
  acceleration/turning (a human hand's jitter; a car's stop-and-go city
  driving) and both land at a smaller, near-identical 4.8% -- consistent
  with the calibrated-rotation model's core assumption (*locally*
  consistent angular velocity/direction) degrading gracefully rather than
  catastrophically as real motion gets less smooth, not a cliff-edge
  failure.

- **Orientation is not a minor add-on -- it's the larger half of the
  compressed size on all three datasets** (60.4% on EuRoC, 57.5% on TUM,
  51.4% on KITTI), starting from a 57.1%-of-raw-bytes share (16 of 28
  bytes per pose) on all three. Its *compression efficiency* relative to
  that raw share is mixed, not uniformly better or worse: on EuRoC and TUM
  (the drone flight and handheld camera -- both continuously reorienting)
  orientation compresses slightly *less* efficiently than position (63.8%
  and 63.6% of its own raw bytes removed, vs. position's 68.4% and 64.1%);
  on KITTI (the vehicle drive, where heading barely changes for long
  straight stretches) orientation compresses *better* than position
  (47.3% of its raw bytes removed vs. position's 33.5%). Either way,
  orientation is consistently a substantial fraction of the total cost,
  not a rounding error next to position -- confirming the Quaternion
  Joint was addressing a real part of the problem, whichever direction
  its relative efficiency happens to land on for a given motion type.

- **No specialized competitor exists to lose to here** (see the intro) --
  unlike the LiDAR-vs-LASzip result, this file can't report "loses
  decisively to the real specialized competitor" because there isn't one
  to test against yet. That is itself the honest caveat: this result says
  CSA's pose codec beats general-purpose compression on real tracking
  data, not that it beats the best anyone has built for this specific
  niche, because nobody has built a dominant specialized one to compare
  against.

- **A verification-methodology note, not a codec bug**: the first pass of
  this benchmark's own round-trip check flagged a false failure on the
  KITTI dataset (a ~1e-6 discrepancy at `SCALE = 1e6`). Traced to one
  source value landing almost exactly on a rounding-tie boundary, where
  Python's `round()` (round-half-to-even, used only by this script's own
  sanity check) and the CLI's `llround()` (round-half-away-from-zero,
  what the actual codec runs) can legitimately disagree by one integer
  unit. Confirmed by comparing against the *quantized* reference instead
  of the raw float, and by `tests/test_main.cpp`'s own extensive
  round-trip checks (which compare integers directly and see none of
  this) already passing. Documented here rather than quietly loosening
  the check without explanation.

## Regenerating these numbers

`bench/real_pose_benchmark.py` regenerates everything in this file. It
needs `scissorc`/`scissorc.exe` built and the three real ground-truth
files downloaded into `bench/_thirdparty/pose_data/` (filenames and exact
source URLs are in the script's module docstring) -- not included in this
repo since they're third-party data, but each is a small (127KB-2.8MB)
plain-text/CSV file, not the multi-GB original dataset releases.
