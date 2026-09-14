// csa/bwt_transform.hpp — the Burrows-Wheeler Transform, block-based.
//
// Unlike the Pantograph Lift and Rod-Joint Transform (both predictive:
// good when nearby samples relate by a small calibrated rule) and the LZ
// matcher (good at exact repeated substrings far apart), the BWT attacks
// a third kind of redundancy: local *context* similarity that doesn't
// require an exact repeat. It permutes the input so that every byte
// migrates next to every other occurrence of the same preceding context,
// turning "this byte tends to follow this context" statistics into long
// runs of identical or near-identical bytes -- which a move-to-front
// pass (bwt_codec.hpp) then turns into mostly-small numbers, and an
// entropy coder compresses well. This is bz2's core technique.
//
// Suffix-array construction is at best O(n log n) and this repo's
// implementation is the simpler, easier-to-verify O(n log^2 n)
// prefix-doubling method (not the linear-time SA-IS algorithm) --
// correctness over cleverness for this one, verified with round-trip
// tests including the classic "banana" textbook example. That cost is
// why this transform operates on fixed-size *blocks*
// (kBwtDefaultBlockSize) rather than the whole file at once, the same
// reason bz2 itself caps its block size (at 900KB) instead of BWT-
// transforming an entire file as one unit.
//
// The construction used here appends one unique sentinel symbol (smaller
// than every real byte) to each block before building its suffix array --
// this guarantees every suffix is distinct (no special-casing needed for
// periodic/constant blocks, e.g. a block of one repeated byte, unlike the
// classical "sort raw cyclic rotations" formulation) at the cost of one
// extra symbol in the transformed block, which is why BwtBlockResult's
// alphabet is 257 values (0 = sentinel, 1..256 = original byte + 1) and
// not a plain byte array.
#pragma once
#include "csa/common.hpp"
#include <vector>

namespace csa {

// 256KB: a suffix array of this size builds in roughly half a second to
// a couple of seconds on typical hardware (this repo's own O(n log^2 n)
// construction, not a highly-tuned linear one) -- big enough that BWT's
// long-range context clustering has real room to work, small enough that
// a multi-MB file's total transform time stays reasonable across its
// blocks.
constexpr size_t kBwtDefaultBlockSize = 256 * 1024;

struct BwtBlockResult {
    std::vector<u16> symbols;  // length original_size+1; sentinel (0) appears exactly once
};

// `block` must be non-empty (empty blocks are handled by the caller,
// bwt_codec.hpp, at the whole-file level).
BwtBlockResult bwt_encode_block(const std::vector<u8>& block);

// Inverse of bwt_encode_block: recovers the original block bytes
// (length symbols.size() - 1) from its BWT symbols.
std::vector<u8> bwt_decode_block(const std::vector<u16>& symbols);

} // namespace csa
