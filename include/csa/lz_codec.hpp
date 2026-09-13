// csa/lz_codec.hpp — entropy coding for an LZ77 token stream (see
// lz_matcher.hpp), producing a self-contained compressed blob.
//
// Literal bytes and the "this position starts a match" flag share one
// order-1 adaptive model over a 257-symbol alphabet (0-255 = literal
// byte, 256 = match). A match's length and distance are each bucketed by
// magnitude class (entropy-coded via their own small adaptive models) with
// the bits distinguishing values within a bucket packed raw (see
// bitpacker.hpp) -- the same length/distance coding strategy DEFLATE and
// LZMA-style codecs use, adapted to this project's range coder instead of
// Huffman coding.
#pragma once
#include "csa/common.hpp"
#include "csa/lz_matcher.hpp"

namespace csa {

// max_chain/nice_length are a pure encoder-side speed/ratio knob (see
// lz_matcher.hpp) -- they don't affect the bitstream format, so decode
// needs no matching parameter.
std::vector<u8> lz_encode(const std::vector<u8>& input,
                          int max_chain = kLzDefaultMaxChain,
                          size_t nice_length = kLzDefaultNiceLength);

// Reads a blob written by lz_encode starting at data[pos], advances pos
// past everything it consumed (so callers can embed this in a larger
// container, the same convention as codec.cpp's other deserialize_*
// helpers), and returns the decoded bytes.
std::vector<u8> lz_decode(const u8* data, size_t size, size_t& pos);

} // namespace csa
