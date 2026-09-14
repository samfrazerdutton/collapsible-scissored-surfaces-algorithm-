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
// Suffix-array construction is done via SA-IS (Nong/Zhang/Chen's
// linear-time construction by induced sorting), trusted through
// exhaustive cross-validation against a much simpler O(n log^2 n)
// prefix-doubling reference implementation (still kept around for
// exactly that purpose -- see bwt_debug_build_suffix_array_reference
// below and test_bwt_sais_matches_reference) rather than through hand-
// checked examples alone, plus the classic "banana" textbook example
// verified by hand against both. Even at O(n), this transform still
// operates on fixed-size *blocks* (kBwtDefaultBlockSize) rather than the
// whole file at once, the same reason bz2 itself caps its block size
// (at 900KB) instead of BWT-transforming an entire file as one unit.
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

// 256KB: big enough that BWT's long-range context clustering has real
// room to work, small enough that a multi-MB file's total transform
// time stays reasonable across its blocks (with the SA-IS construction,
// this is a generous margin rather than a tight necessity -- see
// kBwtMaxInputSize in codec.cpp for the measured cost at real file
// sizes).
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

// Exposed for cross-validation testing only (see
// test_bwt_sais_matches_reference): the O(n log^2 n) prefix-doubling
// reference construction, and the O(n) SA-IS construction
// bwt_encode_block actually uses now, validated against that reference
// on ~150+ randomized cases before ever being trusted to replace it.
// `t`'s last element must be a unique value strictly smaller than every
// other one; `alphabet_size` is one past the largest value that can
// appear in `t`. Not part of the stable public API.
std::vector<int> bwt_debug_build_suffix_array_reference(const std::vector<int>& t);
std::vector<int> bwt_debug_build_suffix_array_sais(const std::vector<int>& t, int alphabet_size);

} // namespace csa
