#include "csa/pose_stream.hpp"

namespace csa {

PoseStreamEncoder::PoseStreamEncoder(std::function<void(const std::vector<u8>&)> on_chunk, size_t chunk_size,
                                      u32 pos_quant_step, u32 pos_resync_interval,
                                      u32 quat_quant_step, u32 quat_resync_interval)
    : on_chunk_(std::move(on_chunk)), chunk_size_(chunk_size == 0 ? 128 : chunk_size),
      pos_quant_step_(pos_quant_step), pos_resync_interval_(pos_resync_interval),
      quat_quant_step_(quat_quant_step), quat_resync_interval_(quat_resync_interval) {}

void PoseStreamEncoder::emit_chunk() {
    bool lossless = (pos_quant_step_ <= 1) && (quat_quant_step_ <= 1);
    std::vector<u8> blob = lossless
        ? compress_pose(buffer_)
        : compress_pose_lossy(buffer_, pos_quant_step_, pos_resync_interval_, quat_quant_step_, quat_resync_interval_);

    std::vector<u8> framed;
    put_u64(framed, (u64)buffer_.size());
    put_u64(framed, (u64)blob.size());
    framed.insert(framed.end(), blob.begin(), blob.end());
    on_chunk_(framed);
    buffer_.clear();
}

void PoseStreamEncoder::push(const Pose& pose) {
    if (finished_) throw std::runtime_error("csa: PoseStreamEncoder::push called after finish()");
    buffer_.push_back(pose);
    if (buffer_.size() >= chunk_size_) emit_chunk();
}

void PoseStreamEncoder::finish() {
    if (finished_) return;
    if (!buffer_.empty()) emit_chunk();
    std::vector<u8> sentinel;
    put_u64(sentinel, 0);
    put_u64(sentinel, 0);
    on_chunk_(sentinel);
    finished_ = true;
}

void PoseStreamDecoder::feed(const u8* data, size_t size) {
    buf_.insert(buf_.end(), data, data + size);
    while (!finished_) {
        if (buf_.size() < 16) return; // not enough bytes for the two u64 header fields yet

        size_t pos = 0;
        u64 pose_count = get_u64(buf_.data(), buf_.size(), pos);
        u64 blob_len = get_u64(buf_.data(), buf_.size(), pos);

        if (pose_count == 0 && blob_len == 0) {
            buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)pos);
            finished_ = true;
            return;
        }
        if (buf_.size() < pos + (size_t)blob_len) return; // chunk not fully arrived yet

        std::vector<u8> blob(buf_.begin() + (std::ptrdiff_t)pos, buf_.begin() + (std::ptrdiff_t)(pos + blob_len));
        buf_.erase(buf_.begin(), buf_.begin() + (std::ptrdiff_t)(pos + blob_len));

        std::vector<Pose> poses = decompress_pose(blob);
        for (const Pose& p : poses) on_pose_(p);
    }
}

} // namespace csa
