// csa/bwt_codec.hpp — full BWT-based compression pipeline: chunk into
// blocks, Burrows-Wheeler Transform each block (bwt_transform.hpp),
// move-to-front encode the result, then adaptive range-code the MTF
// rank stream. This is bz2's core technique (BWT clusters same-context
// bytes into long runs; MTF turns those runs into mostly-small numbers;
// entropy coding compresses the skew), aimed squarely at the kind of
// ordinary-prose redundancy bz2's Burrows-Wheeler stage exploits and
// this codec's other modes don't attack directly.
//
// No dedicated run-length stage for MTF's zero-runs (bz2's RUNA/RUNB
// scheme) -- the adaptive entropy model's own skew-handling captures
// most of that benefit for less implementation risk; see DESIGN.md for
// the honest measured gap this leaves against real bz2.
#pragma once
#include "csa/common.hpp"
#include <vector>

namespace csa {

std::vector<u8> bwt_encode(const std::vector<u8>& input);
std::vector<u8> bwt_decode(const u8* data, size_t size, size_t& pos);

} // namespace csa
