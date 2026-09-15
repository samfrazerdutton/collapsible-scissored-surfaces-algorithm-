"""The actual compression logic for the ROS2 wedge -- deliberately zero
ROS2/rclpy dependency, so it can be (and is) tested directly against the
real built libcsa, without needing a ROS2 install. compressor_node.py and
decompressor_node.py are thin rclpy wrappers around exactly this class;
if something's wrong with the compression itself, it'll show up here,
not buried inside a node that needs a running ROS2 graph to exercise.

Positioning: this is the wedge for the one link that's actually
bandwidth-constrained (a drone's RF downlink, teleoperation over LTE, a
satellite backhaul) -- not a replacement for full-fidelity local logging.
A team's existing `ros2 bag record` / MCAP pipeline keeps recording the
real topic untouched; this sits *alongside* it, specifically on the path
that needs to fit through a narrow pipe. See ADVERSARIAL_BENCHMARK.md for
why that pipe is the actual problem: real ROS2 PoseStamped messages
through MCAP's own default (zstd) lose to CSA by 78-88% on real datasets,
and lose to doing nothing at all on two of three.
"""
try:
    import csa  # if bindings/python is already installed/on PYTHONPATH (a real deployment's normal case)
except ImportError:
    import os
    import sys

    # Dev-repo fallback: reach into this same repo's bindings/python
    # directly, so tests here run against the real built libcsa without
    # requiring a separate install step first.
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "bindings", "python"))
    import csa


class PoseBatchCompressor:
    """Buffers real-valued 6-DOF poses (position in meters, orientation as
    a unit quaternion) and compresses a batch at a time via CSA's
    compress_pose/compress_pose_lossy. scale/qscale fix the fixed-point
    precision (matching the CLI/webapp's own default of 1e6, i.e. micron/
    1e-6-quaternion-unit precision) -- pick a coarser scale if the source
    data doesn't need that much precision; see cli/main.cpp's squeeze for
    the same auto-scale idea done from a file instead of a live stream.
    """

    def __init__(self, scale: int = 1_000_000, qscale: int = 1_000_000,
                 lossy: bool = False, pos_quant_step: int = 1, pos_resync_interval: int = 64,
                 quat_quant_step: int = 1, quat_resync_interval: int = 64):
        self.scale = scale
        self.qscale = qscale
        self.lossy = lossy
        self.pos_quant_step = pos_quant_step
        self.pos_resync_interval = pos_resync_interval
        self.quat_quant_step = quat_quant_step
        self.quat_resync_interval = quat_resync_interval
        self._buffer = []

    def add_pose(self, x: float, y: float, z: float, qw: float, qx: float, qy: float, qz: float) -> None:
        self._buffer.append((x, y, z, qw, qx, qy, qz))

    def __len__(self) -> int:
        return len(self._buffer)

    def flush(self) -> bytes | None:
        """Compresses everything buffered since the last flush and clears
        the buffer. Returns None (not an empty blob) if nothing was
        buffered, so callers can distinguish "nothing to send" from "sent
        an empty batch" without inspecting the bytes."""
        if not self._buffer:
            return None
        poses_int = [
            (
                (round(x * self.scale), round(y * self.scale), round(z * self.scale)),
                (round(qw * self.qscale), round(qx * self.qscale), round(qy * self.qscale), round(qz * self.qscale)),
            )
            for (x, y, z, qw, qx, qy, qz) in self._buffer
        ]
        if self.lossy:
            blob = csa.compress_pose_lossy(
                poses_int, self.pos_quant_step, self.pos_resync_interval,
                self.quat_quant_step, self.quat_resync_interval,
            )
        else:
            blob = csa.compress_pose(poses_int)
        self._buffer = []
        return blob


class PoseBatchDecompressor:
    """The receiving side: turns a compressed batch back into a list of
    (x, y, z, qw, qx, qy, qz) real-valued tuples. scale/qscale must match
    whatever the sender's PoseBatchCompressor used -- there's no
    negotiation here, same as the CLI's --scale being a caller-supplied
    parameter, not something the blob renegotiates."""

    def __init__(self, scale: int = 1_000_000, qscale: int = 1_000_000):
        self.scale = scale
        self.qscale = qscale

    def decode(self, blob: bytes) -> list[tuple[float, float, float, float, float, float, float]]:
        poses_int = csa.decompress_pose(blob)
        return [
            (x / self.scale, y / self.scale, z / self.scale, qw / self.qscale, qx / self.qscale, qy / self.qscale, qz / self.qscale)
            for (x, y, z), (qw, qx, qy, qz) in poses_int
        ]
