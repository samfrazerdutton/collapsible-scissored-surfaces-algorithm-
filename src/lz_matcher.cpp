#include "csa/lz_matcher.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace csa {

namespace {

constexpr int kHashBits = 17;
constexpr size_t kHashSize = size_t(1) << kHashBits;

inline u32 hash4(const u8* p) {
    u32 v = (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
    return (v * 2654435761u) >> (32 - kHashBits);
}

size_t match_length(const u8* a, const u8* b, size_t max_len) {
    size_t len = 0;
    while (len < max_len && a[len] == b[len]) len++;
    return len;
}

} // namespace

// Lazy (one-step lookahead) hash-chain matching, the same technique
// zlib's higher compression levels use: before committing to a match
// found at position i, check whether position i+1 has a strictly longer
// one. If so, emit i as a literal and let the better match at i+1 win
// instead -- a small amount of lookahead that meaningfully improves ratio
// over pure greedy longest-match-first, for a bounded amount of extra
// work (one extra search per position, not a recursive lookahead chain).
//
// Two experiments tried here and reverted, both documented in full in
// DESIGN.md (with the actual code preserved in git history, not just a
// changelog entry):
//
// 1. Selecting matches by an estimated encoded-bit cost (bytes covered
//    per bit, favoring a much closer but slightly shorter match over a
//    farther, marginally longer one) instead of pure greedy-longest. A
//    real, honest, *mixed* result -- helped on some real data, hurt on
//    other realistic data -- traced to the cost estimate being evaluated
//    greedily per candidate with no visibility into how that choice
//    affects the cost of whatever comes after it in the parse.
// 2. A genuine dynamic-programming *optimal* parser (`lz_parse_optimal`,
//    since removed): minimum-cost path over the whole remaining input,
//    using a real Shannon-entropy cost model empirically derived from a
//    baseline lz_parse() of the same input, specifically built to fix
//    experiment 1's "no visibility into what comes after" problem. It
//    was correctly implemented (round-trip correctness held throughout)
//    and, after fixing a real ~13x performance bug (reusing the caller's
//    max_chain at every position instead of the lazy parser's much
//    sparser calling pattern), ran at a reasonable speed -- but measured
//    honestly, it never actually beat the plain lazy parser: identical
//    output size on the 18MB real-source-code corpus (at 2.4x the
//    compress time) and on synthetic templated server-log data. The
//    likely cause: the cost model is derived from the lazy parse's own
//    output statistics, so it's implicitly calibrated toward the lazy
//    parser's own preferences rather than an independent measure of what
//    a genuinely better parse would look like -- a subtler, second-order
//    version of the same "not enough real signal" problem as experiment
//    1, not a coding bug.
std::vector<LzToken> lz_parse(const std::vector<u8>& data, int max_chain, size_t nice_length) {
    std::vector<LzToken> tokens;
    size_t n = data.size();
    if (n == 0) return tokens;

    std::vector<i64> head(kHashSize, -1);
    std::vector<i64> prev(n, -1);
    std::vector<u8> hashed(n, 0); // guards against inserting the same position twice

    auto insert_hash = [&](size_t pos) {
        if (pos + 4 > n || hashed[pos]) return;
        u32 h = hash4(&data[pos]);
        prev[pos] = head[h];
        head[h] = (i64)pos;
        hashed[pos] = 1;
    };

    auto find_match = [&](size_t i, size_t& out_len, size_t& out_dist) {
        out_len = 0;
        out_dist = 0;
        if (i + kLzMinMatch > n) return;
        u32 h = hash4(&data[i]);
        i64 cand = head[h];
        int chain = 0;
        size_t max_possible = n - i;
        while (cand >= 0 && chain < max_chain) {
            size_t len = match_length(&data[(size_t)cand], &data[i], max_possible);
            if (len > out_len) {
                out_len = len;
                out_dist = i - (size_t)cand;
                if (len == max_possible || len >= nice_length) break; // good enough
            }
            cand = prev[(size_t)cand];
            chain++;
        }
    };

    bool have_pending = false;
    size_t pend_len = 0, pend_dist = 0, pend_pos = 0;

    size_t i = 0;
    while (i < n) {
        size_t len = 0, dist = 0;
        // Skip the lookahead search entirely once the pending match is
        // already "nice enough" -- checking whether i+1 is even better
        // is the single most expensive thing this matcher does on highly
        // repetitive input, for a match that's already excellent.
        if (!(have_pending && pend_len >= nice_length)) {
            find_match(i, len, dist);
        }

        if (have_pending) {
            if (len > pend_len) {
                // The match at i beats the one deferred from pend_pos:
                // emit pend_pos as a literal and adopt (i, len, dist) as
                // the new pending match.
                LzToken t;
                t.is_match = false;
                t.literal = data[pend_pos];
                tokens.push_back(t);
                insert_hash(pend_pos);
                pend_len = len;
                pend_dist = dist;
                pend_pos = i;
                i += 1;
            } else {
                // The deferred match is at least as good: commit to it.
                LzToken t;
                t.is_match = true;
                t.length = (u32)pend_len;
                t.distance = (u32)pend_dist;
                tokens.push_back(t);
                for (size_t k = pend_pos; k < pend_pos + pend_len; k++) insert_hash(k);
                i = pend_pos + pend_len;
                have_pending = false;
            }
        } else if (len >= kLzMinMatch) {
            have_pending = true;
            pend_len = len;
            pend_dist = dist;
            pend_pos = i;
            insert_hash(i);
            i += 1;
        } else {
            LzToken t;
            t.is_match = false;
            t.literal = data[i];
            tokens.push_back(t);
            insert_hash(i);
            i += 1;
        }
    }

    if (have_pending) {
        LzToken t;
        t.is_match = true;
        t.length = (u32)pend_len;
        t.distance = (u32)pend_dist;
        tokens.push_back(t);
        // (No need to insert hashes for the trailing match's span: there
        // is no more input left to search against.)
    }

    return tokens;
}

std::vector<u8> lz_tokens_reconstruct(const std::vector<LzToken>& tokens) {
    std::vector<u8> out;
    for (const auto& t : tokens) {
        if (!t.is_match) {
            out.push_back(t.literal);
        } else {
            size_t start = out.size() - t.distance;
            for (u32 k = 0; k < t.length; k++) out.push_back(out[start + k]);
        }
    }
    return out;
}

} // namespace csa
