#include "csa/packet_transport.hpp"
#include "csa/crc32.hpp"

namespace csa {

namespace {

void put_u32_le(std::vector<u8>& out, u32 v) {
    out.push_back((u8)(v & 0xFF));
    out.push_back((u8)((v >> 8) & 0xFF));
    out.push_back((u8)((v >> 16) & 0xFF));
    out.push_back((u8)((v >> 24) & 0xFF));
}

u32 get_u32_le(const u8* data) {
    return (u32)data[0] | ((u32)data[1] << 8) | ((u32)data[2] << 16) | ((u32)data[3] << 24);
}

} // namespace

std::vector<u8> serialize_packet(u32 seq, const std::vector<u8>& payload) {
    // crc32 is computed over seq||payload_len||payload -- the same three
    // fields deserialize_packet cross-checks -- so a corrupted seq or a
    // corrupted length is caught exactly like a corrupted payload byte,
    // not silently trusted just because it happens to parse.
    std::vector<u8> crc_input;
    put_u32_le(crc_input, seq);
    put_u32_le(crc_input, (u32)payload.size());
    crc_input.insert(crc_input.end(), payload.begin(), payload.end());
    u32 crc = crc32(crc_input);

    std::vector<u8> out;
    out.reserve(12 + payload.size());
    put_u32_le(out, seq);
    put_u32_le(out, crc);
    put_u32_le(out, (u32)payload.size());
    out.insert(out.end(), payload.begin(), payload.end());
    return out;
}

ParsedPacket deserialize_packet(const u8* data, size_t size) {
    ParsedPacket p;
    if (size < 12) return p; // too small to even hold the header
    u32 seq = get_u32_le(data);
    u32 stored_crc = get_u32_le(data + 4);
    u32 len = get_u32_le(data + 8);
    if ((size_t)len != size - 12) return p; // declared length disagrees with what actually arrived

    std::vector<u8> crc_input;
    crc_input.reserve(8 + len);
    put_u32_le(crc_input, seq);
    put_u32_le(crc_input, len);
    crc_input.insert(crc_input.end(), data + 12, data + 12 + len);
    if (crc32(crc_input) != stored_crc) return p; // header or payload corrupted

    p.valid = true;
    p.seq = seq;
    p.payload.assign(data + 12, data + 12 + len);
    return p;
}

PacketReassembler::PacketReassembler(std::function<void(u32, const std::vector<u8>&)> on_deliver,
                                      std::function<void(u32, u32)> on_lost,
                                      std::function<void()> on_corrupt,
                                      size_t max_reorder_window)
    : on_deliver_(std::move(on_deliver)), on_lost_(std::move(on_lost)),
      on_corrupt_(std::move(on_corrupt)), max_reorder_window_(max_reorder_window) {}

void PacketReassembler::deliver_and_advance(u32 seq, const std::vector<u8>& payload) {
    on_deliver_(seq, payload);
    next_expected_seq_ = seq + 1;
}

void PacketReassembler::drain_pending() {
    while (true) {
        auto it = pending_.find(next_expected_seq_);
        if (it == pending_.end()) break;
        std::vector<u8> payload = std::move(it->second);
        pending_.erase(it);
        deliver_and_advance(next_expected_seq_, payload);
    }
}

void PacketReassembler::feed(const u8* data, size_t size) {
    ParsedPacket parsed = deserialize_packet(data, size);
    if (!parsed.valid) {
        on_corrupt_();
        return;
    }

    if (parsed.seq < next_expected_seq_) {
        return; // stale duplicate (already delivered, or already written off as lost) -- drop silently
    }

    if (parsed.seq == next_expected_seq_) {
        deliver_and_advance(parsed.seq, parsed.payload);
        drain_pending();
        return;
    }

    // Out of order: parsed.seq > next_expected_seq_. Buffer it first
    // (idempotent if it's a duplicate out-of-order retransmission), then
    // check whether the window is exceeded.
    pending_[parsed.seq] = parsed.payload;

    // If the gap in front of the earliest thing we're actually holding,
    // or the buffer itself, has grown past the window, give up waiting
    // for whatever's still missing -- but only the genuinely missing
    // part. Critically, `target` below is always the smallest seq we
    // actually hold (from pending_, a std::map), never a seq computed
    // from the gap itself, so this loop runs at most once per buffered
    // entry (bounded by max_reorder_window_+1), regardless of how large
    // parsed.seq is -- see the header comment for why an attacker-chosen
    // huge seq must never turn this into a loop over the gap size.
    while (!pending_.empty() &&
           (pending_.size() > max_reorder_window_ ||
            ((u64)pending_.begin()->first - (u64)next_expected_seq_) > (u64)max_reorder_window_)) {
        u32 target = pending_.begin()->first;
        if (target > next_expected_seq_) on_lost_(next_expected_seq_, target - next_expected_seq_);
        std::vector<u8> payload = std::move(pending_.begin()->second);
        pending_.erase(pending_.begin());
        deliver_and_advance(target, payload);
        drain_pending(); // may deliver further already-held, now-contiguous entries too
    }
}

void PacketReassembler::flush() {
    // Same "jump to the smallest seq we actually hold" logic feed()'s
    // window-closure branch uses, just run unconditionally over whatever
    // remains -- there's no more incoming data to wait for.
    while (!pending_.empty()) {
        u32 target = pending_.begin()->first;
        if (target > next_expected_seq_) on_lost_(next_expected_seq_, target - next_expected_seq_);
        std::vector<u8> payload = std::move(pending_.begin()->second);
        pending_.erase(pending_.begin());
        deliver_and_advance(target, payload);
    }
}

} // namespace csa
