# CSA -- the codec layer for physical AI

*Early-stage technical concept. Every number below is measured and
reproducible from the public repo; nothing here is a projection, a
customer claim, or traction data -- there is none yet, and this document
doesn't pretend otherwise. Bracketed fields are for you to fill in.*

## The problem

Autonomous drone fleets, humanoid robots, AV perception stacks, and
teleoperated machines all generate 6-DOF pose streams and LiDAR point
clouds faster than the links carrying them can handle -- drone RF,
degraded 5G, satellite backhaul, teleoperation over LTE. The tools this
industry already uses for it (rosbag2/MCAP with zstd compression, generic
byte-level compressors) were built for arbitrary data, not for the
specific kinematic structure inside pose/point-cloud streams -- and it
shows: real ROS2 `PoseStamped` messages through MCAP's default (zstd)
compression measured **78-88% larger than a purpose-built codec on the
same real data, and larger than doing nothing at all on two of three real
datasets tested.** There is no MP3 or H.264 for robot telemetry -- no
incumbent codec standard for the data shape physical AI actually produces.

## What exists today (built, measured, public)

- **A real C++ compression codec**, not a wrapper around an existing one:
  calibrated rotation prediction for point trajectories (Rod-Joint
  Transform), calibrated delta-quaternion prediction for orientation
  (Quaternion Joint), per-block adaptive precision, a parallel entropy
  coder (interleaved rANS) with both a native multi-threaded path and a
  WebGPU port.
- **Honest, adversarial, reproducible benchmarks** -- including the losses.
  Beats gzip/bz2/lzma/zstd/brotli and real ROS2/MCAP tooling on real 6-DOF
  pose data (KITTI, EuRoC, TUM ground truth) by 63-88%. Loses decisively to
  the specialized LASzip codec on real airborne LiDAR geometry -- reported
  plainly, not hidden. See `ADVERSARIAL_BENCHMARK.md`, `REAL_POSE_BENCHMARK.md`,
  `REAL_GEO_BENCHMARK.md`.
- **Three working demos, zero backend**: a WASM build running the real
  C++ core in-browser with live WebGPU-accelerated decode; a zero-backend
  static page (hostable free on GitHub Pages) with real file upload/
  download; a CLI (`scissorc squeeze`/`unsqueeze`) that auto-detects
  whether a file is a 2D/3D trajectory, a 6-DOF pose stream, or general
  bytes, with no schema to define.
- **A ROS2 integration wedge** (`ros2_wedge/`) -- drop-in nodes for one
  pose topic on the constrained link specifically, alongside (not
  replacing) a team's existing rosbag2/MCAP recording.
- **Four language bindings** (Python, Rust, C#, Go) over a stable C ABI.

## Why now

"Physical AI" -- humanoid robots, drone fleets, AV perception, teleoperation
-- is the category attracting real capital right now, and its data
infrastructure is still built on tools designed for arbitrary bytes, not
kinematics. Generic compression is a commodity nobody pays for (zstd is
free and good enough for most things); domain-aware compression for this
specific data shape, under real bandwidth constraints, is not yet
commoditized the same way -- there's no incumbent standard.

## Go-to-market: wedge, not platform

1. **Distribution**: the honest, adversarial benchmark (already built) is
   the free distribution channel -- publish it, let engineers verify it
   themselves.
2. **Adoption unit**: the ROS2 wedge (already built) -- one topic, one
   robot, an afternoon, no sales conversation required to try it.
3. **Monetization**: open-core. Core codec, CLI, demos, bindings, the
   single-topic wedge, and the honest benchmarks stay open -- that's the
   trust and distribution engine. Paid: per-fleet adaptive-precision
   tuning, multi-robot/fleet-scale orchestration, a managed backhaul/
   observability layer, and certified builds for safety-critical/embedded
   targets (none of which exist yet -- see `OPEN_CORE_SCOPE.md` for the
   full reasoning, including the honest risk that this category could
   commoditize the same way generic compression already has).

## Honest open questions

- No live-link testing yet -- every number above is a real file-size
  comparison, not a test over real degraded RF/5G/satellite hardware with
  real packet loss.
- No embedded/RTOS validation -- nothing here has been ported to or tested
  on real drone/robot flight hardware.
- No customers, no revenue, no team beyond [FOUNDER NAME(S)] as of this
  writing.

## Ask

[STAGE / AMOUNT / USE OF FUNDS -- fill in]

## Contact

[NAME] -- [EMAIL] -- [REPO LINK]
