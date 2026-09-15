// See csa/rans_coder.hpp for the design rationale. The core single-lane
// algorithm below is the standard byte-oriented rANS structure (as
// popularized by Fabian Giesen's public-domain rans_byte.h reference):
// state renormalizes by whole bytes, encoding processes symbols in
// reverse (building the output backwards so it reads correctly forward),
// decoding processes forward. Getting this exactly right matters more
// than being clever about it -- see tests/test_main.cpp's
// test_rans_coder_round_trip for the exhaustive cross-check this leans
// on before any ratio or speed claim is trusted.
#include "csa/rans_coder.hpp"
#include <algorithm>
#include <array>
#include <thread>

namespace csa {

namespace {

constexpr u32 kRansL = 1u << 23; // lower bound of the renormalization interval

struct RansTable {
    int scale_bits = 14;
    u32 M = 1u << 14;
    std::array<u32, 256> freq{};
    std::array<u32, 257> cum_freq{};
    std::vector<u16> slot_to_symbol; // size M
};

RansTable build_table(const std::array<u32, 256>& freq, int scale_bits) {
    RansTable t;
    t.scale_bits = scale_bits;
    t.M = 1u << scale_bits;
    t.freq = freq;
    t.cum_freq[0] = 0;
    for (int s = 0; s < 256; s++) t.cum_freq[s + 1] = t.cum_freq[s] + t.freq[s];
    t.slot_to_symbol.assign(t.M, 0);
    for (int s = 0; s < 256; s++)
        for (u32 slot = t.cum_freq[s]; slot < t.cum_freq[s + 1]; slot++)
            t.slot_to_symbol[slot] = (u16)s;
    return t;
}

// Quantizes a raw byte histogram to frequencies summing to exactly M,
// giving every symbol that actually occurs at least frequency 1 (so it's
// always representable), and nudging the single largest-frequency
// symbol(s) one unit at a time to absorb whatever rounding error is left
// -- a small, cheap, exact-terminating adjustment (the error is bounded
// by the number of distinct symbols, well under scale_bits precision).
std::array<u32, 256> normalize_histogram(const std::array<u64, 256>& hist, u64 total, u32 M) {
    std::array<u32, 256> freq{};
    if (total == 0) return freq;
    u64 assigned = 0;
    for (int s = 0; s < 256; s++) {
        if (hist[s] == 0) continue;
        u64 f = (hist[s] * M) / total;
        if (f == 0) f = 1;
        freq[s] = (u32)f;
        assigned += f;
    }
    i64 diff = (i64)M - (i64)assigned;
    while (diff != 0) {
        int best = -1;
        for (int s = 0; s < 256; s++) {
            if (freq[s] == 0) continue;
            if (diff > 0) { if (best < 0 || freq[s] > freq[best]) best = s; }
            else { if (freq[s] > 1 && (best < 0 || freq[s] > freq[best])) best = s; }
        }
        if (best < 0) break; // shouldn't happen for any real histogram at these scale_bits
        if (diff > 0) { freq[best]++; diff--; }
        else { freq[best]--; diff++; }
    }
    return freq;
}

// Encodes n symbols (in forward order) into a self-contained rANS
// lane -- output is already in correct forward-read order (built by
// writing backwards into a generously-sized scratch buffer, then
// returning just the written suffix).
std::vector<u8> rans_encode_lane(const u8* data, size_t n, const RansTable& t) {
    std::vector<u8> buf(n * 2 + 64); // generous upper bound; real output is close to entropy-optimal
    size_t pos = buf.size();
    u32 x = kRansL;

    for (size_t i = n; i-- > 0; ) {
        u8 s = data[i];
        u32 freq = t.freq[s];
        u32 start = t.cum_freq[s];
        u32 x_max = ((kRansL >> t.scale_bits) << 8) * freq;
        while (x >= x_max) {
            buf[--pos] = (u8)(x & 0xFF);
            x >>= 8;
        }
        x = ((x / freq) << t.scale_bits) + (x % freq) + start;
    }
    // Flush the final state, 4 bytes, so that reading forward from the
    // front of the lane gives little-endian x (matches the decode-side
    // initial read below).
    buf[--pos] = (u8)(x >> 24);
    buf[--pos] = (u8)(x >> 16);
    buf[--pos] = (u8)(x >> 8);
    buf[--pos] = (u8)(x >> 0);

    return std::vector<u8>(buf.begin() + (std::ptrdiff_t)pos, buf.end());
}

std::vector<u8> rans_decode_lane(const u8* data, size_t n_symbols, const RansTable& t) {
    std::vector<u8> out(n_symbols);
    const u8* ptr = data;
    u32 x = (u32)ptr[0] | ((u32)ptr[1] << 8) | ((u32)ptr[2] << 16) | ((u32)ptr[3] << 24);
    ptr += 4;

    for (size_t i = 0; i < n_symbols; i++) {
        u32 slot = x & (t.M - 1);
        u8 s = (u8)t.slot_to_symbol[slot];
        out[i] = s;
        u32 freq = t.freq[s];
        u32 start = t.cum_freq[s];
        x = freq * (x >> t.scale_bits) + slot - start;
        while (x < kRansL) x = (x << 8) | (u32)(*ptr++);
    }
    return out;
}

} // namespace

std::vector<u8> encode_interleaved_rans(const std::vector<u8>& data, int num_lanes, int scale_bits) {
    if (num_lanes <= 0) num_lanes = 1;
    if (scale_bits <= 0 || scale_bits > 16) scale_bits = 14;
    u32 M = 1u << scale_bits;

    std::vector<u8> out;
    put_u64(out, (u64)data.size());
    put_u32(out, (u32)num_lanes);
    put_u32(out, (u32)scale_bits);
    if (data.empty()) return out;

    std::array<u64, 256> hist{};
    for (u8 b : data) hist[b]++;
    std::array<u32, 256> freq = normalize_histogram(hist, data.size(), M);
    for (int s = 0; s < 256; s++) write_varint(out, freq[s]);

    RansTable table = build_table(freq, scale_bits);

    // Split into num_lanes contiguous, as-equal-as-possible chunks. If
    // there are more lanes than symbols, the smallest lanes simply end
    // up with 0 symbols (a valid, exercised case -- see the round-trip test).
    size_t n = data.size();
    size_t base = n / (size_t)num_lanes, rem = n % (size_t)num_lanes;
    std::vector<size_t> lane_starts((size_t)num_lanes + 1);
    lane_starts[0] = 0;
    for (int i = 0; i < num_lanes; i++)
        lane_starts[i + 1] = lane_starts[i] + base + ((size_t)i < rem ? 1 : 0);

    std::vector<std::vector<u8>> lane_out((size_t)num_lanes);
    {
        std::vector<std::thread> threads;
        threads.reserve((size_t)num_lanes);
        for (int i = 0; i < num_lanes; i++) {
            threads.emplace_back([&, i]() {
                size_t s = lane_starts[i], e = lane_starts[i + 1];
                lane_out[i] = rans_encode_lane(data.data() + s, e - s, table);
            });
        }
        for (auto& th : threads) th.join();
    }

    for (int i = 0; i < num_lanes; i++) {
        put_u64(out, lane_starts[i + 1] - lane_starts[i]);
        put_u64(out, (u64)lane_out[i].size());
        out.insert(out.end(), lane_out[i].begin(), lane_out[i].end());
    }
    return out;
}

std::vector<u8> decode_interleaved_rans(const std::vector<u8>& blob) {
    size_t pos = 0;
    u64 total_len = get_u64(blob.data(), blob.size(), pos);
    u32 num_lanes = get_u32(blob.data(), blob.size(), pos);
    u32 scale_bits = get_u32(blob.data(), blob.size(), pos);
    if (total_len == 0) return {};

    std::array<u32, 256> freq{};
    for (int s = 0; s < 256; s++) freq[s] = (u32)read_varint(blob.data(), blob.size(), pos);
    RansTable table = build_table(freq, (int)scale_bits);

    std::vector<size_t> lane_symcount(num_lanes), lane_bytelen(num_lanes), lane_byteoffset(num_lanes), lane_out_offset(num_lanes);
    size_t out_offset = 0;
    for (u32 i = 0; i < num_lanes; i++) {
        u64 symcount = get_u64(blob.data(), blob.size(), pos);
        u64 bytelen = get_u64(blob.data(), blob.size(), pos);
        lane_symcount[i] = (size_t)symcount;
        lane_bytelen[i] = (size_t)bytelen;
        lane_byteoffset[i] = pos;
        lane_out_offset[i] = out_offset;
        out_offset += (size_t)symcount;
        pos += (size_t)bytelen;
        if (pos > blob.size()) throw std::runtime_error("csa: truncated interleaved-rans lane");
    }
    if (out_offset != (size_t)total_len) throw std::runtime_error("csa: interleaved-rans lane lengths don't sum to total_len");

    std::vector<u8> out(total_len);
    {
        std::vector<std::thread> threads;
        threads.reserve(num_lanes);
        for (u32 i = 0; i < num_lanes; i++) {
            threads.emplace_back([&, i]() {
                if (lane_symcount[i] == 0) return;
                std::vector<u8> lane_result = rans_decode_lane(blob.data() + lane_byteoffset[i], lane_symcount[i], table);
                std::copy(lane_result.begin(), lane_result.end(), out.begin() + (std::ptrdiff_t)lane_out_offset[i]);
            });
        }
        for (auto& th : threads) th.join();
    }
    return out;
}

} // namespace csa
