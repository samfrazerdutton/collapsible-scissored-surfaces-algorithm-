// csa/common.hpp — shared basic types and helpers for the Collapsible Scissored
// Surfaces compression algorithm (CSA).
#pragma once
#include <cstdint>
#include <cstddef>
#include <vector>
#include <stdexcept>

namespace csa {

using i8  = int8_t;
using u8  = uint8_t;
using i16 = int16_t;
using u16 = uint16_t;
using i32 = int32_t;
using u32 = uint32_t;
using i64 = int64_t;
using u64 = uint64_t;

// Fixed-point scale used for calibrated ratio/rotation parameters (Q16.16).
constexpr i32 kFixedShift = 16;
constexpr i64 kFixedOne   = i64(1) << kFixedShift;

inline i64 fixed_mul_round(i64 value, i64 fixed_ratio) {
    // value * (fixed_ratio / 2^16), rounded to nearest. Plain int64 math
    // (no __int128 — MSVC's cl.exe doesn't provide it); callers keep
    // |value| and |fixed_ratio| small enough (well under 2^31 each) that
    // the product never approaches the int64 range.
    i64 prod = value * fixed_ratio;
    i64 half = i64(1) << (kFixedShift - 1);
    if (prod >= 0) prod += half; else prod -= half;
    return prod >> kFixedShift;
}

// Rounds d/q to the nearest integer (round-half-away-from-zero), for
// q >= 1. q == 1 returns d unchanged exactly, which is what makes q=1
// mean "lossless" rather than "lossy with a suspiciously small step" --
// shared by every lossy mode in this codebase (Rod-Joint 2D/3D, and the
// Pantograph Lift) so they all agree on this rounding convention.
inline i64 quant_round_div(i64 d, i64 q) {
    if (q <= 1) return d;
    return (d >= 0) ? (d + q / 2) / q : -((-d + q / 2) / q);
}

// ---- Zigzag encoding: maps signed ints to unsigned so small-magnitude
// residuals (positive or negative) become small unsigned values, which the
// entropy coder can then exploit. ----
// Both encoders below compute the textbook zigzag formula
// (v << 1) ^ (v >> N) entirely in the *unsigned* domain: left-shifting a
// negative signed value is undefined behavior (found for real by
// UBSan -- see docs/SANITIZERS.md -- not a theoretical concern), and
// right-shifting one is merely implementation-defined rather than
// portably specified. Casting to unsigned first and reconstructing the
// arithmetic-shift-style sign mask via `0u - (uv >> (bits-1))` (an
// unsigned shift, always well-defined, extracting just the sign bit)
// produces the exact same bit pattern on every two's-complement
// platform this project targets, with no UB and no implementation-
// defined step anywhere in the computation.
inline u32 zigzag_encode32(i32 v) {
    u32 uv = (u32)v;
    u32 mask = (u32)0 - (uv >> 31);
    return (uv << 1) ^ mask;
}
inline i32 zigzag_decode32(u32 v) {
    return (i32)((v >> 1) ^ (~(v & 1) + 1));
}
inline u64 zigzag_encode64(i64 v) {
    u64 uv = (u64)v;
    u64 mask = (u64)0 - (uv >> 63);
    return (uv << 1) ^ mask;
}
inline i64 zigzag_decode64(u64 v) {
    return (i64)((v >> 1) ^ (~(v & 1) + 1));
}

// ---- Little-endian varint (LEB128-style) for zigzag-encoded values. ----
inline void write_varint(std::vector<u8>& out, u64 v) {
    while (v >= 0x80) {
        out.push_back(u8(v) | 0x80);
        v >>= 7;
    }
    out.push_back(u8(v));
}
inline u64 read_varint(const u8* data, size_t size, size_t& pos) {
    u64 result = 0;
    int shift = 0;
    while (true) {
        if (pos >= size) throw std::runtime_error("varint: truncated stream");
        u8 b = data[pos++];
        result |= (u64)(b & 0x7F) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
        if (shift > 63) throw std::runtime_error("varint: too long");
    }
    return result;
}

// ---- Small fixed-width helpers for header serialization (little-endian). ----
inline void put_u32(std::vector<u8>& out, u32 v) {
    out.push_back(u8(v & 0xFF));
    out.push_back(u8((v >> 8) & 0xFF));
    out.push_back(u8((v >> 16) & 0xFF));
    out.push_back(u8((v >> 24) & 0xFF));
}
inline void put_u64(std::vector<u8>& out, u64 v) {
    for (int i = 0; i < 8; i++) out.push_back(u8((v >> (8 * i)) & 0xFF));
}
inline void put_i32(std::vector<u8>& out, i32 v) { put_u32(out, (u32)v); }

inline u32 get_u32(const u8* data, size_t size, size_t& pos) {
    if (pos + 4 > size) throw std::runtime_error("header: truncated u32");
    u32 v = (u32)data[pos] | ((u32)data[pos+1] << 8) |
            ((u32)data[pos+2] << 16) | ((u32)data[pos+3] << 24);
    pos += 4;
    return v;
}
inline u64 get_u64(const u8* data, size_t size, size_t& pos) {
    if (pos + 8 > size) throw std::runtime_error("header: truncated u64");
    u64 v = 0;
    for (int i = 0; i < 8; i++) v |= (u64)data[pos + i] << (8 * i);
    pos += 8;
    return v;
}
inline i32 get_i32(const u8* data, size_t size, size_t& pos) {
    return (i32)get_u32(data, size, pos);
}

} // namespace csa
