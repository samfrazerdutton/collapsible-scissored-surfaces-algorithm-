"""ROS2 node: the receiving side of compressor_node.py. Subscribes to a
csa_bridge_msgs/msg/CompressedPoseBatch topic (as would arrive over the
constrained link on the ground-station/base-station end) and
republishes each pose in the batch as a real geometry_msgs/msg/PoseStamped,
so anything downstream (RViz, a planner, a logger) sees an ordinary pose
stream and doesn't need to know CSA was ever involved.

NOT execution-tested in this environment (no ROS2 install here) -- see
compressor_node.py's docstring for the same caveat and what *is* tested
(csa_bridge/core.py, directly, against the real built libcsa).

Usage:
    ros2 run csa_bridge decompressor_node --ros-args \\
        -p input_topic:=/pose/csa_compressed -p output_topic:=/pose/restored
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from csa_bridge_msgs.msg import CompressedPoseBatch

from csa_bridge.core import PoseBatchDecompressor


class CsaPoseDecompressorNode(Node):
    def __init__(self):
        super().__init__("csa_pose_decompressor")

        self.declare_parameter("input_topic", "/pose/csa_compressed")
        self.declare_parameter("output_topic", "/pose/restored")

        input_topic = self.get_parameter("input_topic").value
        output_topic = self.get_parameter("output_topic").value

        self._sub = self.create_subscription(CompressedPoseBatch, input_topic, self._on_batch, 10)
        self._pub = self.create_publisher(PoseStamped, output_topic, 10)

        self.get_logger().info(f"csa_pose_decompressor: {input_topic} -> {output_topic}")

    def _on_batch(self, msg: CompressedPoseBatch):
        decoder = PoseBatchDecompressor(scale=msg.scale, qscale=msg.qscale)
        try:
            poses = decoder.decode(bytes(msg.data))
        except Exception as e:  # noqa: BLE001 -- a malformed/corrupt batch should not crash the node
            self.get_logger().error(f"failed to decode batch from {msg.source_topic}: {e}")
            return

        if len(poses) != msg.pose_count:
            self.get_logger().warning(
                f"decoded {len(poses)} poses but batch header said {msg.pose_count} -- publishing what decoded anyway"
            )

        for (x, y, z, qw, qx, qy, qz) in poses:
            out = PoseStamped()
            out.header.stamp = msg.header.stamp
            out.header.frame_id = msg.header.frame_id
            out.pose.position.x, out.pose.position.y, out.pose.position.z = x, y, z
            out.pose.orientation.w, out.pose.orientation.x = qw, qx
            out.pose.orientation.y, out.pose.orientation.z = qy, qz
            self._pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = CsaPoseDecompressorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
