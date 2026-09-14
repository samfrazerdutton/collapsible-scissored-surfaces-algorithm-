// Demo: a simulated live GPS/IMU feed compressed and decoded incrementally
// via PoseStreamEncoder/PoseStreamDecoder, as opposed to compress_pose's
// batch API. Not a permanent CLI feature -- a standalone demo proving the
// streaming API's real properties: bounded memory (only ever holds one
// chunk's worth of poses, never the whole flight), incremental output
// (bytes are produced as data arrives, not only at the end), and genuine
// incremental decode (a "receiver" reconstructs poses chunk by chunk as
// bytes arrive, fed in small pieces that don't line up with chunk
// boundaries, simulating a real byte stream rather than assuming each
// read() happens to hand over a whole chunk).
//
// Build manually against the built csa_core static library, e.g. from an
// MSVC dev prompt:
//   cl /nologo /EHsc /O2 /MD /std:c++17 /I include
//      bench/pose_stream_demo.cpp /Fe:build/pose_stream_demo.exe
//      /link build/csa_core.lib
#include "csa/pose_stream.hpp"
#include <cmath>
#include <cstdio>

using namespace csa;

namespace {

// Simulates a drone's GPS+IMU feed: circling and climbing (position),
// yaw tracking the circular heading with small real-world noise
// (orientation) -- the same realistic shape class as this session's other
// synthetic pose tests, generated one sample at a time rather than as a
// pre-built vector, to genuinely mimic a live sensor feed.
class SimulatedFeed {
public:
    bool next(Pose& out) {
        if (i_ >= total_) return false;
        double t = i_ * 0.05;
        i32 px = (i32)std::lround(2000 * std::cos(t));
        i32 py = (i32)std::lround(2000 * std::sin(t));
        z_ += 4.0 + (next_rand() - 0.5) * 0.5;
        i32 pz = (i32)std::lround(z_);

        const double qscale = 1 << 20;
        out.position = {px, py, pz};
        out.orientation = {(i32)std::lround(qw_ * qscale), (i32)std::lround(qx_ * qscale),
                            (i32)std::lround(qy_ * qscale), (i32)std::lround(qz_ * qscale)};

        double deg = 2.0 + (next_rand() - 0.5) * 0.3;
        double half = (deg * 3.14159265358979323846 / 180.0) / 2.0;
        double dqw = std::cos(half), dqz = std::sin(half);
        double nw = qw_ * dqw - qz_ * dqz;
        double nx = qx_ * dqw + qy_ * dqz;
        double ny = qy_ * dqw - qx_ * dqz;
        double nz = qw_ * dqz + qz_ * dqw;
        qw_ = nw; qx_ = nx; qy_ = ny; qz_ = nz;

        i_++;
        return true;
    }

    explicit SimulatedFeed(int total) : total_(total) {}

private:
    double next_rand() { seed_ = seed_ * 1664525u + 1013904223u; return (double)(seed_ >> 8) / (double)(1u << 24); }
    int i_ = 0, total_;
    unsigned int seed_ = 12345;
    double z_ = 0.0;
    double qw_ = 1, qx_ = 0, qy_ = 0, qz_ = 0;
};

} // namespace

int main(int argc, char** argv) {
    int total_poses = (argc > 1) ? std::atoi(argv[1]) : 2000;
    size_t chunk_size = (argc > 2) ? (size_t)std::atoi(argv[2]) : 128;

    std::printf("Simulating a live %d-sample GPS/IMU feed, chunk_size=%zu\n\n", total_poses, chunk_size);

    // "Receiver": decodes poses as their containing chunk arrives, fed in
    // small (61-byte) pieces that deliberately don't align to chunk
    // boundaries, to prove the decoder handles real, arbitrary delivery.
    std::vector<Pose> received;
    PoseStreamDecoder decoder([&](const Pose& p) { received.push_back(p); });

    size_t chunks_emitted = 0;
    size_t total_wire_bytes = 0;

    PoseStreamEncoder encoder([&](const std::vector<u8>& chunk) {
        chunks_emitted++;
        total_wire_bytes += chunk.size();
        std::printf("  chunk %zu emitted: %zu bytes (running total on the wire: %zu bytes)\n",
                     chunks_emitted, chunk.size(), total_wire_bytes);
        // Feed the "transport" in small, non-chunk-aligned pieces.
        size_t pos = 0;
        const size_t piece = 61;
        while (pos < chunk.size()) {
            size_t n = std::min(piece, chunk.size() - pos);
            decoder.feed(chunk.data() + pos, n);
            pos += n;
        }
    }, chunk_size);

    SimulatedFeed feed(total_poses);
    std::vector<Pose> sent;
    Pose p;
    while (feed.next(p)) {
        sent.push_back(p);
        encoder.push(p); // encoder never holds more than one chunk's worth of poses
    }
    encoder.finish();

    std::printf("\nStream finished: %zu poses sent, %zu received, decoder finished=%s\n",
                 sent.size(), received.size(), decoder.finished() ? "yes" : "no");

    bool match = received.size() == sent.size();
    for (size_t i = 0; match && i < sent.size(); i++) {
        const Pose& a = sent[i]; const Pose& b = received[i];
        if (a.position.x != b.position.x || a.position.y != b.position.y || a.position.z != b.position.z ||
            a.orientation.w != b.orientation.w || a.orientation.x != b.orientation.x ||
            a.orientation.y != b.orientation.y || a.orientation.z != b.orientation.z) match = false;
    }
    std::printf("Round-trip: %s\n\n", match ? "PASS (every pose reconstructed exactly)" : "FAIL");

    // Honest comparison against the batch codec on the identical data --
    // chunking has a real, measured ratio cost, reported here rather than
    // only in the test suite.
    std::vector<u8> batch_blob = compress_pose(sent);
    std::printf("Streamed total: %zu bytes across %zu chunks\n", total_wire_bytes, chunks_emitted);
    std::printf("Batch compress_pose on the same data: %zu bytes\n", batch_blob.size());
    std::printf("Streaming overhead: %.1f%% larger than batch, for real bounded memory + incremental output\n",
                 100.0 * ((double)total_wire_bytes / (double)batch_blob.size() - 1.0));
    return match ? 0 : 1;
}
