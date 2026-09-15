"""ROS2 node: subscribes to a geometry_msgs/msg/PoseStamped topic, batches
poses via PoseBatchCompressor (see core.py -- that's the part actually
tested against the real libcsa without needing ROS2), and publishes a
csa_bridge_msgs/msg/CompressedPoseBatch every `batch_size` poses or every
`flush_period_sec`, whichever comes first. Intended to run on the
transmitting side of a bandwidth-constrained link (a drone, a
teleoperated robot) alongside -- not instead of -- a normal
`ros2 bag record` of the same PoseStamped topic for full-fidelity local
logging.

NOT execution-tested in this environment: this repo's dev machine does
not have ROS2 installed, so this file has been written carefully against
documented rclpy APIs and cross-checked by reading (not run against a
live ROS2 graph). core.py, which holds all the actual compression logic
this file calls into, IS tested -- see ros2_wedge/test/test_core.py. If
you run this against a real ROS2 install and something's off in the
rclpy plumbing specifically (not the compression), that's the part to
suspect first.

Usage (with a real ROS2 install, from a workspace containing this package):
    ros2 run csa_bridge compressor_node --ros-args \\
        -p pose_topic:=/drone/pose -p batch_size:=64 -p flush_period_sec:=0.5
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from csa_bridge_msgs.msg import CompressedPoseBatch

from csa_bridge.core import PoseBatchCompressor


class CsaPoseCompressorNode(Node):
    def __init__(self):
        super().__init__("csa_pose_compressor")

        self.declare_parameter("pose_topic", "/pose")
        self.declare_parameter("output_topic", "/pose/csa_compressed")
        self.declare_parameter("batch_size", 64)
        self.declare_parameter("flush_period_sec", 1.0)
        self.declare_parameter("lossy", False)
        self.declare_parameter("pos_quant_step", 1)
        self.declare_parameter("pos_resync_interval", 64)
        self.declare_parameter("quat_quant_step", 1)
        self.declare_parameter("quat_resync_interval", 64)
        self.declare_parameter("scale", 1_000_000)
        self.declare_parameter("qscale", 1_000_000)

        pose_topic = self.get_parameter("pose_topic").value
        output_topic = self.get_parameter("output_topic").value
        self._batch_size = self.get_parameter("batch_size").value
        self._source_topic = pose_topic
        self._frame_id = ""  # filled in from the first message seen

        self._compressor = PoseBatchCompressor(
            scale=self.get_parameter("scale").value,
            qscale=self.get_parameter("qscale").value,
            lossy=self.get_parameter("lossy").value,
            pos_quant_step=self.get_parameter("pos_quant_step").value,
            pos_resync_interval=self.get_parameter("pos_resync_interval").value,
            quat_quant_step=self.get_parameter("quat_quant_step").value,
            quat_resync_interval=self.get_parameter("quat_resync_interval").value,
        )

        self._sub = self.create_subscription(PoseStamped, pose_topic, self._on_pose, 10)
        self._pub = self.create_publisher(CompressedPoseBatch, output_topic, 10)
        flush_period = self.get_parameter("flush_period_sec").value
        self._timer = self.create_timer(flush_period, self._on_timer_flush)

        self.get_logger().info(
            f"csa_pose_compressor: {pose_topic} -> {output_topic} "
            f"(batch_size={self._batch_size}, flush_period={flush_period}s, lossy={self._compressor.lossy})"
        )

    def _on_pose(self, msg: PoseStamped):
        self._frame_id = msg.header.frame_id
        p, o = msg.pose.position, msg.pose.orientation
        self._compressor.add_pose(p.x, p.y, p.z, o.w, o.x, o.y, o.z)
        if len(self._compressor) >= self._batch_size:
            self._flush()

    def _on_timer_flush(self):
        if len(self._compressor) > 0:
            self._flush()

    def _flush(self):
        pose_count = len(self._compressor)
        blob = self._compressor.flush()
        if blob is None:
            return
        out = CompressedPoseBatch()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self._frame_id
        out.source_topic = self._source_topic
        out.pose_count = pose_count
        out.lossy = self._compressor.lossy
        out.scale = self._compressor.scale
        out.qscale = self._compressor.qscale
        out.data = list(blob)
        self._pub.publish(out)
        self.get_logger().debug(f"flushed {pose_count} poses -> {len(blob)} compressed bytes")


def main(args=None):
    rclpy.init(args=args)
    node = CsaPoseCompressorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
