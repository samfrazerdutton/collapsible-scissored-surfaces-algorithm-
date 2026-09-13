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
// the same hash, walked (up to max_chain candidates) to find the longest
// match, with lazy (one-step lookahead) parsing. Every position covered
// by an accepted match still gets inserted into the hash chains, so a
// later match can reference into the middle of an earlier one -- this
// matters a lot for highly repetitive data.
//
// max_chain/nice_length are a real speed-vs-ratio knob, the same
// tradeoff every production LZ compressor exposes (gzip -1..-9, zstd
// -1..-22): a smaller max_chain searches fewer candidates per position,
// and a smaller nice_length gives up on already-good matches sooner
// (skipping the lazy lookahead too) -- both trade a small amount of
// ratio for a large amount of speed on highly repetitive input, where
// hash chains get very long. See REAL_CORPUS_BENCHMARK.md for measured
// numbers: on 18MB of real source code, max_chain=1024/nice_length=huge
// took ~30s at 0.6 MB/s; max_chain=64/nice_length=64 took ~1.7s at
// ~11 MB/s for a ~2% ratio cost.
#pragma once
#include "csa/common.hpp"

namespace csa {

constexpr u32 kLzMinMatch = 4;

// Sensible defaults, tuned against REAL_CORPUS_BENCHMARK.md's real
// source-code corpus and USE_CASES.md's realistic log/JSON/CSV data
// together (not just one or the other): still beats gzip on most
// realistic text, while keeping worst-case time on highly repetitive
// input bounded to a small multiple of gzip/lzma's.
constexpr int kLzDefaultMaxChain = 128;
constexpr size_t kLzDefaultNiceLength = 128;

struct LzToken {
    bool is_match = false;
    u8 literal = 0;    // valid when !is_match
    u32 length = 0;    // valid when is_match: actual match length (>= kLzMinMatch)
    u32 distance = 0;  // valid when is_match: backward distance, >= 1
};

std::vector<LzToken> lz_parse(const std::vector<u8>& data,
                               int max_chain = kLzDefaultMaxChain,
                               size_t nice_length = kLzDefaultNiceLength);

// Reference reconstruction used to unit-test lz_parse in isolation from
// the entropy-coding layer: replays a token stream with no entropy coding
// involved and must recover the original bytes exactly.
std::vector<u8> lz_tokens_reconstruct(const std::vector<LzToken>& tokens);

} // namespace csa
