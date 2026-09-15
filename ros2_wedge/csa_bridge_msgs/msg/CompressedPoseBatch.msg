# A batch of 6-DOF poses compressed with CSA (compress_pose/compress_pose_lossy).
# Not an MCAP/rosbag2 replacement -- MCAP's own compression is fixed to
# zstd/lz4/none at the wire format level, not pluggable, so this rides as
# its own topic on the specific constrained link (drone RF, teleop over
# LTE, satellite backhaul) that real ROS2 PoseStamped + MCAP's default
# zstd loses badly on (see ADVERSARIAL_BENCHMARK.md). A team's existing
# `ros2 bag record` of the source PoseStamped topic is untouched by this
# -- it keeps recording full-fidelity, this is purely the squeezed
# version that actually fits through the narrow pipe.

std_msgs/Header header      # stamp = time this batch was flushed; frame_id = the source topic's frame_id
string source_topic         # the PoseStamped topic this batch came from
uint32 pose_count           # how many poses are in this batch (decoder cross-checks this against the decoded count)
bool lossy                  # whether compress_pose_lossy (true) or compress_pose (false) was used
int64 scale                 # fixed-point scale used for position (must match on both ends)
int64 qscale                # fixed-point scale used for orientation (must match on both ends)
uint8[] data                 # the actual compressed bytes -- csa.compress_pose(_lossy)'s return value, unmodified
