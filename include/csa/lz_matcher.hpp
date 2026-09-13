// csa/lz_matcher.hpp — LZ77-style dictionary match finder.
//
// The Pantograph Lift and Rod-Joint transforms are predictive: good when
// consecutive samples/rods relate by a small calibrated rule. Ordinary
// text and structured files instead have *repeated substrings* far apart
// (a word, a log-line template, a repeated header) -- a fundamentally
// different kind of redundancy that a predictive transform cannot exploit
// but a dictionary matcher is built for. This is the missing piece for
// competing with gzip/bz2/lzma on that kind of data; it does not replace
// the geometric modes, which remain the literal point of this repo.
//
// This is an unbounded-window (whole-buffer) hash-chain matcher: a
// position's 4-byte hash indexes a chain of every earlier position with
// the same hash, walked (up to a bounded depth) to find the longest
// match. Matches are greedy (no lookahead/lazy matching yet -- see
// DESIGN.md's future work), but every position covered by an accepted
// match still gets inserted into the hash chains, so a later match can
// reference into the middle of an earlier one -- this matters a lot for
// highly repetitive data.
#pragma once
#include "csa/common.hpp"

namespace csa {

constexpr u32 kLzMinMatch = 4;

struct LzToken {
    bool is_match = false;
    u8 literal = 0;    // valid when !is_match
    u32 length = 0;    // valid when is_match: actual match length (>= kLzMinMatch)
    u32 distance = 0;  // valid when is_match: backward distance, >= 1
};

std::vector<LzToken> lz_parse(const std::vector<u8>& data);

// Reference reconstruction used to unit-test lz_parse in isolation from
// the entropy-coding layer: replays a token stream with no entropy coding
// involved and must recover the original bytes exactly.
std::vector<u8> lz_tokens_reconstruct(const std::vector<LzToken>& tokens);

} // namespace csa
