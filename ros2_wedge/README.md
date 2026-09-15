# csa_bridge -- the CSA wedge for ROS2

The adoption unit, not a platform: a team drops this into their workspace,
points it at *one* pose topic on their most bandwidth-constrained link
(a drone's RF downlink, teleoperation over LTE, a satellite backhaul), and
either it wins on their real traffic or it doesn't. See
`ADVERSARIAL_BENCHMARK.md` in the repo root for why that link specifically
is where this matters: real ROS2 `geometry_msgs/msg/PoseStamped` messages
through MCAP's own default (zstd) compression lose to CSA by 78-88% on
every real dataset measured there, and lose to *doing nothing at all* on
two of three -- not because zstd is a bad compressor, but because a
PoseStamped message's float64 + repeated header/timestamp/frame_id framing
is already ~2.5-3x bigger than it needs to be before any compression
happens, and general-purpose entropy coding can't see that consecutive
quaternions differ by a small, smoothly-varying delta rotation.

This is **not** a rosbag2/MCAP replacement, and it doesn't try to be. Your
existing `ros2 bag record` of the source topic keeps recording full-fidelity
locally, untouched. `csa_bridge` sits *alongside* it, specifically on the
narrow pipe: `compressor_node` batches and compresses the pose stream with
the real CSA codec, publishes a small `CompressedPoseBatch` message that
actually fits through that link; `decompressor_node` on the other end turns
it back into an ordinary `PoseStamped` stream, so anything downstream (RViz,
a planner, a logger) never needs to know CSA was involved at all.

## What's real, tested, and running right now

`csa_bridge/csa_bridge/core.py` -- `PoseBatchCompressor`/`PoseBatchDecompressor`
-- has **zero ROS2 dependency** and calls directly into the real, already-
built `libcsa` through this repo's existing `bindings/python/csa.py` ctypes
wrapper (the same binding this repo's own Python tests use, not a new
reimplementation). This is the part that actually matters -- if the
compression is wrong, it's wrong here, not hidden inside a node that needs
a live ROS2 graph to even exercise it. It's tested right now, with no
ROS2 install required:

```
python ros2_wedge/csa_bridge/test/test_core.py
```

```
  lossless: 500 poses, naive float64 packing~=28000 bytes -> 2184 bytes (92.2% smaller), max round-trip error=5.00e-07
  lossy: 1002 bytes, max summed position error=1.10e-04m
12 checks, 0 failures
```

## What's written but NOT execution-tested here

`compressor_node.py` and `decompressor_node.py` are thin `rclpy` wrappers
around exactly the class above -- real subscriptions, a real timer-based
flush, real parameter declarations, following documented `rclpy`/
`ament_python` conventions. **This development machine does not have ROS2
installed**, so these two files (and the `csa_bridge_msgs` package's
`CompressedPoseBatch.msg`/`CMakeLists.txt`) have been written carefully and
cross-checked by reading, not run against a live ROS2 graph or built with
`colcon`. If something's off when you actually build this, it's much more
likely to be in the `rclpy` plumbing in these two files than in the
compression logic, which is independently proven.

## Layout

```
ros2_wedge/
  csa_bridge_msgs/            # ament_cmake package: the CompressedPoseBatch.msg interface
    package.xml
    CMakeLists.txt
    msg/CompressedPoseBatch.msg
  csa_bridge/                 # ament_python package: the actual nodes
    package.xml
    setup.py / setup.cfg
    resource/csa_bridge
    csa_bridge/
      core.py                 # zero-ROS2-dependency compression logic -- tested
      compressor_node.py       # rclpy wrapper -- written, not execution-tested here
      decompressor_node.py     # rclpy wrapper -- written, not execution-tested here
    test/test_core.py         # runs right now, no ROS2 needed
```

## Building it for real (with an actual ROS2 install)

```bash
# from a ROS2 workspace's src/ directory
git clone <this repo> csa   # or symlink ros2_wedge's two packages in directly
cd ../..
colcon build --packages-select csa_bridge_msgs csa_bridge
source install/setup.bash

# transmitting side (e.g. on the drone)
ros2 run csa_bridge compressor_node --ros-args \
    -p pose_topic:=/drone/pose -p batch_size:=64 -p flush_period_sec:=0.5

# receiving side (e.g. at the ground station)
ros2 run csa_bridge decompressor_node --ros-args \
    -p input_topic:=/pose/csa_compressed -p output_topic:=/pose/restored
```

`csa_bridge`'s Python code needs `libcsa` built and either on `CSA_LIB_PATH`
or discoverable next to this repo's `build/` directory -- see
`bindings/python/csa.py`'s own docstring.

## Honest open questions this wedge doesn't answer yet

- **Real link testing.** Every number in `ADVERSARIAL_BENCHMARK.md` is a
  file-size comparison on real recorded datasets, not a test over an actual
  degraded link (real LoRa/5G/satellite hardware, real packet loss, real
  retransmission behavior). The demo pages' "simulated transmission time"
  is explicitly labeled illustrative for the same reason.
- **Batching latency tradeoff.** Bigger batches compress better (CSA's
  per-block calibration has more data to work with) but add latency before
  a pose is available on the other end. `batch_size`/`flush_period_sec` are
  exposed as parameters precisely because there's no single right answer
  here -- it depends on the application's actual latency budget, which this
  package doesn't know.
- **Point clouds.** This wedge only covers 6-DOF pose streams
  (`compress_pose`). A `sensor_msgs/msg/PointCloud2` equivalent using
  `compress_geo3d` would be the natural next package, not built here.
