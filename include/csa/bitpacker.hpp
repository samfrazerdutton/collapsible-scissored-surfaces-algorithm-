// csa/bitpacker.hpp — a minimal raw (non-entropy-coded) bit packer.
//
// Used by the LZ codec (lz_codec.hpp) for match length/distance "extra
// bits": once a value's magnitude class (bucket) has been entropy-coded,
// the bits distinguishing it from other values in that bucket are close
// to uniformly distributed by construction, so entropy coding them gains
// essentially nothing over packing them raw -- this is the same design
// DEFLATE/LZMA-style codecs use for length/distance extra bits.
#pragma once
#include "csa/common.hpp"

namespace csa {

class BitWriter {
public:
    explicit BitWriter(std::vector<u8>& out) : out_(out) {}

    // Writes the low `nbits` bits of `value`, most-significant-bit first.
    void write_bits(u64 value, int nbits) {
        for (int i = nbits - 1; i >= 0; i--) {
            cur_ = (u8)((cur_ << 1) | ((value >> i) & 1));
            if (++nbits_in_cur_ == 8) {
                out_.push_back(cur_);
                cur_ = 0;
                nbits_in_cur_ = 0;
            }
        }
    }

    // Pads the final partial byte with zero bits and flushes it.
    void flush() {
        if (nbits_in_cur_ > 0) {
            cur_ = (u8)(cur_ << (8 - nbits_in_cur_));
            out_.push_back(cur_);
            cur_ = 0;
            nbits_in_cur_ = 0;
        }
    }

private:
    std::vector<u8>& out_;
    u8 cur_ = 0;
    int nbits_in_cur_ = 0;
};

class BitReader {
public:
    BitReader(const u8* data, size_t size) : data_(data), size_(size) {}

    u64 read_bits(int nbits) {
        u64 v = 0;
        for (int i = 0; i < nbits; i++) {
            if (nbits_in_cur_ == 0) {
                cur_ = (pos_ < size_) ? data_[pos_++] : 0;
                nbits_in_cur_ = 8;
            }
            v = (v << 1) | ((cur_ >> 7) & 1);
            cur_ = (u8)(cur_ << 1);
            nbits_in_cur_--;
        }
        return v;
    }

private:
    const u8* data_;
    size_t size_;
    size_t pos_ = 0;
    u8 cur_ = 0;
    int nbits_in_cur_ = 0;
};

} // namespace csa
