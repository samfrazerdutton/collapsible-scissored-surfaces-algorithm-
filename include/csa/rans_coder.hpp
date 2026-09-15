// csa/rans_coder.hpp — interleaved rANS (range Asymmetric Numeral System)
// entropy coder: an alternative to range_coder.hpp's adaptive order-1
// range coder, built specifically for parallelism rather than ratio.
//
// range_coder.hpp is fundamentally sequential: each symbol's encoding
// depends on the *adapted* frequency state left behind by every symbol
// before it (see its own header comment for why that was the deliberate
// choice over rANS -- correctness-simplicity over parallelism). This
// file is the other side of that tradeoff, built for the cases where
// parallelism matters more than squeezing out the last bit of ratio:
// input is split into independent "lanes," each encoded/decoded with its
// own rANS state but sharing ONE static (not adaptive) frequency table
// built once from the whole input -- exactly the "interleaved rANS"
// technique real GPU compressors (nvcomp and friends) use, where each
// lane maps to one GPU thread/warp with no cross-lane dependency at all.
//
// The core single-lane algorithm is the standard byte-oriented rANS
// design (state renormalizes by emitting/consuming whole bytes, table
// built by quantizing a histogram to a power-of-two total) -- the same
// structure used by every public rANS reference implementation (e.g.
// Fabian Giesen's rans_byte.h), not a novel scheme, because getting a
// *novel* entropy coder bit-exact is exactly the kind of subtly-wrong
// territory range_coder.hpp's own header comment already warns about.
// What's new here is the interleaving/lane framing on top of it, and the
// honest measurement of what it costs and what it's worth.
//
// Because the table is static (fit once over the whole input, not
// updated per symbol), this is an order-0 coder, not order-1 like
// range_coder.hpp -- a real, measured ratio cost on most realistic data,
// not a free upgrade. See DESIGN.md for the actual numbers.
#pragma once
#include "csa/common.hpp"
#include <vector>

namespace csa {

// Encodes `data` as `num_lanes` independent rANS-coded lanes sharing one
// static order-0 frequency table fit over the whole input. num_lanes <= 0
// is treated as 1. Lanes are encoded (and, in decode_interleaved_rans,
// decoded) using std::thread across up to num_lanes threads -- a real,
// immediately-usable CPU parallelism win from the same independence that
// makes this GPU-friendly, not just a design property on paper.
std::vector<u8> encode_interleaved_rans(const std::vector<u8>& data, int num_lanes = 4, int scale_bits = 14);

// Decodes a buffer produced by encode_interleaved_rans. Throws
// std::runtime_error on a malformed/truncated buffer.
std::vector<u8> decode_interleaved_rans(const std::vector<u8>& blob);

} // namespace csa
