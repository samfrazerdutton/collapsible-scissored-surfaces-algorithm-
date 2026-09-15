# Why not just zstd?

This is the first question anyone technical will ask, so here's the
honest answer up front, not buried in a benchmark appendix: **for most
files, zstd is exactly what you should use, and CSA will lose to it.**
`REAL_CORPUS_BENCHMARK.md` shows CSA losing to zstd -19 on 18MB of real
C++ source code -- both on size *and* speed at once. That's not a fluke
or a weak test case; it's real source code, and zstd (and lzma) are
genuinely excellent general-purpose compressors with years of production
tuning behind them. If your data is "arbitrary bytes" -- logs, source
code, JSON, most files most people compress most of the time -- reach
for zstd. CSA is not trying to replace it there and does not claim to.

## So when does CSA actually win, and why

On one specific, narrow data shape: **6-DOF pose streams and point-cloud
trajectories**, where the numbers being compressed aren't arbitrary --
they're samples of a continuous physical motion (a drone flying, a robot's
end effector moving, a vehicle driving). On that data, real measured
numbers:

- Beats gzip/bz2/lzma/zstd/brotli by 43-89% on three real ground-truth
  trajectory datasets (KITTI, EuRoC, TUM) -- see `REAL_POSE_BENCHMARK.md`.
- Beats real ROS2 `PoseStamped` messages through MCAP's own default zstd
  compression by 78-88% on the same datasets -- see
  `ADVERSARIAL_BENCHMARK.md`.

**Why zstd can't just close this gap by being a better compressor**: it's
not a compressor-strength problem, it's a representation problem. zstd
(and every general-purpose compressor) finds redundancy that's visible
*in the byte stream it's given* -- repeated substrings, statistical
skew, that kind of thing. A sequence of raw quaternions from a smoothly
rotating body doesn't have much of that: each float64 looks close to
random relative to its neighbors at the byte level, even though the
*rotation itself* is highly predictable (a body in flight doesn't
instantaneously reverse its angular velocity). CSA's Quaternion Joint
transform computes an actual calibrated delta-rotation per block and
encodes the *residual* -- which is small and genuinely low-entropy,
because the physical prediction was good -- and only then hands that
residual to an entropy coder. No amount of zstd window size or
compression level can manufacture that predictive step after the fact,
because the structure it needs (this is a rotation, not an arbitrary
7-tuple of floats) was already thrown away by the time the bytes reach
zstd. This is the same reason FLAC beats zstd on audio and H.264 beats
zstd on video: domain-specific prediction ahead of generic entropy
coding, not a better generic entropy coder.

## Where CSA loses even inside its own claimed domain

This isn't a universal win even for kinematic data, and the docs say so
plainly:

- **Point clouds specifically** (not trajectories): CSA loses decisively
  to the specialized LASzip codec on a real airborne LiDAR scan -- 24.2%
  larger, and 25-75x slower. LASzip's per-point predictor is built around
  how airborne scanners actually sweep a surface (scan order, return
  number, GPS-time locality); CSA's model (calibrated rotation between
  consecutive edge vectors) fits a genuinely curving *path* well and an
  approximately-random-order point cloud poorly. See
  `REAL_GEO_BENCHMARK.md`'s full writeup, including *why* the model
  applies to trajectories but not scan-order point clouds specifically.
- **Draco**, Google's point-cloud codec, beats CSA on ratio at its
  roughest quantization setting (coarser than useful for most real LiDAR
  work) -- see `ADVERSARIAL_BENCHMARK.md`.

## The actual decision

| Your data | Reach for |
|---|---|
| Logs, source code, JSON, arbitrary files | zstd (or lzma if you can afford the time for a bit more ratio) |
| A 6-DOF pose/trajectory stream (drone flight, robot odometry, camera path, vehicle path) | CSA -- this is the case it's built for and measurably wins on |
| A LiDAR/point-cloud scan | LASzip if it's airborne/terrestrial LAS-shaped data; Draco if you need mesh support or a broader ecosystem; CSA is not currently the right choice here despite point clouds sounding adjacent to its other strengths |
| You don't know which shape your data is | `scissorc squeeze` auto-detects and falls back to general compression (which, per the top of this doc, will usually lose to zstd -- so if it detects "general," that's itself useful information: your data doesn't have the structure this project targets) |

If you're still unsure after reading this, that's a completely reasonable
place to land -- the answer genuinely is "it depends on your data's
shape," not "always CSA." Every number cited above is reproducible;
`bench/` has the scripts.
