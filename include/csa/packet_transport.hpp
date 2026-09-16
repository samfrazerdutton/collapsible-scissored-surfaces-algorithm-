// csa/packet_transport.hpp — a real, generic, loss/corruption/reorder-
// tolerant packet transport layer, built to sit underneath
// pose_stream.hpp's chunked encoder/decoder (or any other length-prefixed
// producer) when the real transport is packet-oriented and unreliable
// (UDP, a lossy radio link, an unordered message queue) rather than a
// reliable, ordered byte stream (TCP, a local pipe) -- pose_stream.hpp's
// own PoseStreamDecoder::feed() assumes exactly the reliable/ordered case
// and is unmodified by this file; this is a separate layer underneath it,
// not a replacement.
//
// Wire format per packet: [u32 seq][u32 crc32][u32 payload_len][payload].
// crc32 covers seq and payload_len as well as the payload itself (see
// serialize_packet/deserialize_packet below), so a corrupted header is
// caught by the same check as a corrupted payload, not silently trusted.
// Each feed() call takes one already-framed datagram (the size the
// caller received from its actual transport) -- this layer does not
// itself solve byte-stream framing, since pose_stream.hpp's own
// length-prefix scheme already does that for the reliable-transport case.
#pragma once
#include "csa/common.hpp"
#include <functional>
#include <map>
#include <vector>

namespace csa {

std::vector<u8> serialize_packet(u32 seq, const std::vector<u8>& payload);

struct ParsedPacket {
    bool valid = false; // false if truncated, size-mismatched, or CRC-mismatched
    u32 seq = 0;
    std::vector<u8> payload;
};
ParsedPacket deserialize_packet(const u8* data, size_t size);

// Accepts packets in *any* arrival order (not necessarily seq order),
// buffers out-of-order arrivals up to `max_reorder_window` entries (or
// `max_reorder_window` sequence numbers ahead of what's still awaited,
// whichever is smaller), and delivers payloads to `on_deliver` strictly
// in seq order once each is either received or given up on.
//
// Deliberately bounded, not just "buffer until it arrives": an
// adversarial or badly out-of-order sender can send a `seq` far larger
// than what this receiver is still waiting for, and closing that gap one
// integer at a time (calling on_lost once per missing seq, or advancing
// next_expected_seq_ one at a time) would be the same class of CPU-time-
// amplification bug this project's fuzzing pass found and fixed
// elsewhere (see docs/SANITIZERS.md, bug #7) -- a single packet claiming
// a huge seq could otherwise force an unbounded loop. Instead, once the
// window is exceeded, the receiver jumps directly to the smallest seq it
// actually holds (never to the gap size itself), reporting only the
// genuinely-missing range in front of it as one on_lost call -- bounded
// by the number of buffered packets (at most max_reorder_window_+1
// iterations), never by how large an attacker-chosen seq is.
class PacketReassembler {
public:
    // on_deliver(seq, payload): called once per seq, strictly in order,
    // for every packet that either arrived and validated or was skipped
    // because it was never going to (see on_lost below) -- skipped seqs
    // simply never appear here.
    // on_lost(first_seq, count): called when [first_seq, first_seq+count)
    // are given up on as a contiguous block, always before any later seq
    // in that gap is delivered.
    // on_corrupt(): called when a fed datagram fails validation (CRC
    // mismatch or a truncated/size-mismatched header) -- its seq isn't
    // trustworthy (the corruption could be in the seq field itself), so
    // it carries no arguments; the packet is dropped, not queued.
    explicit PacketReassembler(std::function<void(u32, const std::vector<u8>&)> on_deliver,
                                std::function<void(u32, u32)> on_lost,
                                std::function<void()> on_corrupt,
                                size_t max_reorder_window = 64);

    // Feed one already-framed datagram (exactly the bytes one real
    // transport receive call returned). Arrival order need not match seq
    // order.
    void feed(const u8* data, size_t size);
    void feed(const std::vector<u8>& data) { feed(data.data(), data.size()); }

    // Call exactly once when the caller knows no further packets are
    // coming (mirroring PoseStreamEncoder::finish()'s role on the sender
    // side): delivers everything still buffered, in seq order, reporting
    // any remaining gaps as lost first. Without this, a handful of
    // genuinely-received packets sitting in the reorder buffer at the
    // very end of a stream could otherwise wait forever for a window
    // closure that no further packet will ever arrive to trigger.
    void flush();

private:
    void deliver_and_advance(u32 seq, const std::vector<u8>& payload);
    void drain_pending();

    std::function<void(u32, const std::vector<u8>&)> on_deliver_;
    std::function<void(u32, u32)> on_lost_;
    std::function<void()> on_corrupt_;
    size_t max_reorder_window_;
    u32 next_expected_seq_ = 0;
    std::map<u32, std::vector<u8>> pending_;
};

} // namespace csa
