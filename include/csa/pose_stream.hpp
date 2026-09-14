// csa/pose_stream.hpp — genuinely incremental encode/decode for a live
// 6-DOF pose stream (VR/AR tracking, drone/robot odometry, a GPS+IMU
// feed), as opposed to compress_pose/decompress_pose's batch API, which
// needs the entire sequence in memory before producing any output.
//
// Design: buffer at most `chunk_size` poses at a time (not the whole
// stream), and compress each chunk independently via the existing,
// already-tested compress_pose()/decompress_pose() the moment it fills --
// emitting a complete, self-contained, length-prefixed chunk immediately.
// This gives real bounded memory (O(chunk_size), not O(stream length))
// and real incremental output/decode (a consumer sees data as it's
// produced, not only after the whole stream ends), without re-deriving
// any of the calibration/serialization internals: every chunk is just an
// ordinary compress_pose() call on a short sequence.
//
// The cost of this simplicity is real and worth stating plainly: treating
// every chunk as an independent sequence means every chunk pays the same
// "no history yet" cost the batch codec's own first block already pays
// (rods before a candidate lag predict from zero instead of an earlier
// sample) -- instead of paying it once per stream, a chunked stream pays
// it once per chunk. Measured impact on a realistic pose stream is in
// DESIGN.md; it is a real, bounded ratio cost, not free.
//
// Wire format per chunk: [u64 pose_count][u64 blob_len][blob_len bytes of
// compress_pose()/compress_pose_lossy() output]. A final sentinel chunk
// with pose_count == 0 and blob_len == 0 marks the end of the stream, so
// a decoder never needs to know the total length in advance.
#pragma once
#include "csa/codec.hpp"
#include "csa/quaternion_joint.hpp"
#include <functional>
#include <stdexcept>
#include <vector>

namespace csa {

class PoseStreamEncoder {
public:
    // on_chunk is invoked synchronously, once per completed chunk (and
    // once more at finish() with the end-of-stream sentinel) -- e.g.
    // write it to a socket, a file, or append it to an in-memory buffer
    // simulating a live feed. quant_step <= 1 on either axis is lossless
    // for that axis, exactly like compress_pose_lossy.
    explicit PoseStreamEncoder(std::function<void(const std::vector<u8>&)> on_chunk,
                                size_t chunk_size = 128,
                                u32 pos_quant_step = 1, u32 pos_resync_interval = 0,
                                u32 quat_quant_step = 1, u32 quat_resync_interval = 0);

    // Buffers one pose; emits a complete chunk via on_chunk the moment
    // chunk_size poses have accumulated.
    void push(const Pose& pose);

    // Flushes any partial trailing chunk (if the stream doesn't end on
    // an exact chunk boundary) and emits the end-of-stream sentinel.
    // Must be called exactly once, after the last push(); push() after
    // finish() throws.
    void finish();

private:
    void emit_chunk();

    std::function<void(const std::vector<u8>&)> on_chunk_;
    size_t chunk_size_;
    u32 pos_quant_step_, pos_resync_interval_, quat_quant_step_, quat_resync_interval_;
    std::vector<Pose> buffer_;
    bool finished_ = false;
};

class PoseStreamDecoder {
public:
    // on_pose is invoked once per pose, in order, as soon as the chunk
    // containing it has fully arrived and been decoded -- never waits
    // for the whole stream.
    explicit PoseStreamDecoder(std::function<void(const Pose&)> on_pose) : on_pose_(std::move(on_pose)) {}

    // Feed newly-arrived bytes, in any chunking the transport happens to
    // deliver (a single byte, a partial chunk, several chunks at once).
    // Decodes and dispatches every fully-arrived chunk it can; buffers
    // only the undelivered tail of the next one.
    void feed(const u8* data, size_t size);
    void feed(const std::vector<u8>& data) { feed(data.data(), data.size()); }

    // True once the end-of-stream sentinel chunk has been received and processed.
    bool finished() const { return finished_; }

private:
    std::function<void(const Pose&)> on_pose_;
    std::vector<u8> buf_;
    bool finished_ = false;
};

} // namespace csa
