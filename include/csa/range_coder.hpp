// csa/range_coder.hpp — byte-oriented carryless range coder (Subbotin style)
// with an adaptive order-1 frequency model backed by per-context Fenwick
// (binary indexed) trees. This is the entropy-coding backend applied to the
// residual streams produced by the Pantograph Lift / Rod-Joint transforms.
#pragma once
#include "csa/common.hpp"
#include <array>
#include <memory>

namespace csa {

// ---- Fenwick tree over a fixed 256-symbol alphabet, used as one adaptive
// frequency table. Supports point increment, prefix-sum query, and
// "find index at which cumulative frequency crosses a target" (needed to
// decode a symbol from a code value). ----
class FenwickFreq {
public:
    static constexpr int kSymbols = 256;
    static constexpr u32 kIncrement = 32;
    static constexpr u32 kMaxTotal = 1u << 15;

    FenwickFreq() { reset(); }

    void reset() {
        tree_.fill(0);
        total_ = 0;
        for (int s = 0; s < kSymbols; s++) add(s, 1);
        total_ = kSymbols; // each of the 256 symbols now has frequency 1
    }

    u32 total() const { return total_; }

    // Cumulative frequency of symbols [0, sym).
    u32 cumfreq(int sym) const {
        u32 sum = 0;
        for (int i = sym; i > 0; i -= i & (-i)) sum += tree_[i - 1];
        return sum;
    }

    u32 freq_of(int sym) const { return cumfreq(sym + 1) - cumfreq(sym); }

    // Given a target cumulative value in [0, total), find the symbol whose
    // [cumfreq(sym), cumfreq(sym)+freq(sym)) range contains it, and report
    // that range via lo/f.
    int find(u32 target, u32& lo, u32& f) const {
        int idx = 0;
        u32 rem = target;
        int logn = 8; // log2(256)
        for (int pw = 1 << logn; pw > 0; pw >>= 1) {
            int next = idx + pw;
            if (next <= kSymbols && (u32)tree_[next - 1] <= rem) {
                idx = next;
                rem -= tree_[next - 1];
            }
        }
        int sym = idx; // idx is the count of symbols fully consumed
        lo = target - rem;
        f = freq_of(sym);
        return sym;
    }

    void update(int sym) {
        add(sym, (i32)kIncrement);
        total_ += kIncrement;
        if (total_ > kMaxTotal) rescale();
    }

private:
    void add(int sym, i32 delta) {
        for (int i = sym + 1; i <= kSymbols; i += i & (-i)) tree_[i - 1] += delta;
    }

    void rescale() {
        // Halve every symbol's frequency (floor, min 1), rebuild the tree
        // from scratch via prefix differences.
        std::array<u32, kSymbols> freqs{};
        for (int s = 0; s < kSymbols; s++) {
            u32 f = freq_of(s);
            freqs[s] = (f > 1) ? (f >> 1) : 1;
        }
        tree_.fill(0);
        total_ = 0;
        for (int s = 0; s < kSymbols; s++) {
            add(s, (i32)freqs[s]);
            total_ += freqs[s];
        }
    }

    std::array<i32, kSymbols> tree_{};
    u32 total_ = 0;
};

// Order-1 model: one FenwickFreq per preceding-byte context (256 contexts).
class Order1Model {
public:
    Order1Model() : ctx_(256) {}
    FenwickFreq& context(int prev_byte) { return ctx_[(u8)prev_byte]; }
private:
    std::vector<FenwickFreq> ctx_;
};

// ---- Range coder core (32-bit, byte renormalization). ----
class RangeEncoder {
public:
    explicit RangeEncoder(std::vector<u8>& out) : out_(out) {}

    void encode(u32 cumfreq, u32 freq, u32 totfreq) {
        range_ /= totfreq;
        low_ += (u64)cumfreq * range_;
        range_ *= freq;
        normalize();
    }

    void finish() {
        for (int i = 0; i < 4; i++) {
            out_.push_back(u8(low_ >> 24));
            low_ <<= 8;
        }
    }

private:
    void normalize() {
        // Carryless range coder: emit top byte whenever the range has
        // shrunk enough that top-byte is settled, or force renorm when
        // range gets too small to keep precision (Schindler/Subbotin trick).
        while ((low_ ^ (low_ + range_)) < kTop ||
               (range_ < kBot && ((range_ = (u32)(-(i64)low_) & (kBot - 1)), true))) {
            out_.push_back(u8(low_ >> 24));
            low_ <<= 8;
            range_ <<= 8;
        }
    }

    static constexpr u32 kTop = 1u << 24;
    static constexpr u32 kBot = 1u << 16;
    std::vector<u8>& out_;
    u32 low_ = 0;
    u32 range_ = 0xFFFFFFFFu;
};

class RangeDecoder {
public:
    RangeDecoder(const u8* data, size_t size) : data_(data), size_(size) {
        for (int i = 0; i < 4; i++) code_ = (code_ << 8) | next_byte();
    }

    u32 get_freq(u32 totfreq) {
        range_ /= totfreq;
        return (code_ - low_) / range_;
    }

    void decode(u32 cumfreq, u32 freq) {
        low_ += cumfreq * range_;
        range_ *= freq;
        normalize();
    }

private:
    u8 next_byte() { return pos_ < size_ ? data_[pos_++] : 0; }

    void normalize() {
        while ((low_ ^ (low_ + range_)) < kTop ||
               (range_ < kBot && ((range_ = (u32)(-(i64)low_) & (kBot - 1)), true))) {
            code_ = (code_ << 8) | next_byte();
            low_ <<= 8;
            range_ <<= 8;
        }
    }

    static constexpr u32 kTop = 1u << 24;
    static constexpr u32 kBot = 1u << 16;
    const u8* data_;
    size_t size_;
    size_t pos_ = 0;
    u32 low_ = 0;
    u32 range_ = 0xFFFFFFFFu;
    u32 code_ = 0;
};

// High-level helpers: encode/decode a byte buffer with the adaptive
// order-1 model. `decode_bytes` needs the exact output length up front
// (stored separately in the container header).
std::vector<u8> range_encode_bytes(const std::vector<u8>& input);
std::vector<u8> range_decode_bytes(const u8* data, size_t size, size_t output_length);

} // namespace csa
