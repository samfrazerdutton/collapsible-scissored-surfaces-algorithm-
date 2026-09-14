#include "csa/bwt_transform.hpp"
#include <algorithm>
#include <array>
#include <climits>

namespace csa {

namespace {

// Prefix-doubling suffix array construction (Manber-Myers style):
// O(log n) rounds, each an O(n log n) sort keyed by the current
// (rank[i], rank[i+k]) pair, giving O(n log^2 n) overall. Every element
// of `t` must be non-negative except sentinel-style "smaller than
// everything" markers, which this function itself never introduces --
// callers (bwt_encode_block) are responsible for making all suffixes of
// `t` distinct (via a unique minimum-valued sentinel) so no tie ever
// needs an arbitrary tiebreak.
std::vector<int> build_suffix_array(const std::vector<int>& t) {
    int n = (int)t.size();
    std::vector<int> sa(n), rnk(n), tmp(n);
    for (int i = 0; i < n; i++) { sa[i] = i; rnk[i] = t[i]; }

    for (int k = 1; k < n; k *= 2) {
        auto second_key = [&](int idx) { return (idx + k < n) ? rnk[idx + k] : INT_MIN; };
        auto cmp = [&](int a, int b) {
            if (rnk[a] != rnk[b]) return rnk[a] < rnk[b];
            return second_key(a) < second_key(b);
        };
        std::sort(sa.begin(), sa.end(), cmp);

        tmp[sa[0]] = 0;
        for (int i = 1; i < n; i++)
            tmp[sa[i]] = tmp[sa[i - 1]] + (cmp(sa[i - 1], sa[i]) ? 1 : 0);
        rnk.swap(tmp);

        if (rnk[sa[n - 1]] == n - 1) break; // every rank distinct: fully sorted
    }
    return sa;
}

} // namespace

BwtBlockResult bwt_encode_block(const std::vector<u8>& block) {
    BwtBlockResult r;
    int n = (int)block.size();
    if (n == 0) return r;

    // t[0..n-1] = block bytes shifted up by one (1..256), t[n] = 0 (the
    // sentinel, guaranteed smaller than every real byte) -- so every
    // suffix of t is distinct regardless of how repetitive `block` is.
    std::vector<int> t(n + 1);
    for (int i = 0; i < n; i++) t[i] = (int)block[i] + 1;
    t[n] = 0;

    std::vector<int> sa = build_suffix_array(t);

    int m = n + 1;
    r.symbols.resize(m);
    for (int i = 0; i < m; i++) {
        int pos = sa[i];
        // "Previous" character in t, treating t as cyclic over its own
        // length m for this one lookup -- the standard BWT-via-suffix-
        // array-with-sentinel construction. When pos == 0 (the row whose
        // suffix is the entire string t), the previous character wraps
        // to t[m-1], which is exactly the sentinel itself.
        int prev = (pos == 0) ? t[m - 1] : t[pos - 1];
        r.symbols[i] = (u16)prev;
    }
    return r;
}

std::vector<u8> bwt_decode_block(const std::vector<u16>& symbols) {
    int m = (int)symbols.size();
    if (m == 0) return {};

    constexpr int kAlphabet = 257; // 0 = sentinel, 1..256 = byte + 1
    std::array<int, kAlphabet> count{};
    for (u16 s : symbols) count[s]++;

    // C[c] = number of symbols strictly less than c -- the start offset
    // of value c's block in sorted order.
    std::array<int, kAlphabet> C{};
    int total = 0;
    for (int c = 0; c < kAlphabet; c++) { C[c] = total; total += count[c]; }

    // next[i] = the row whose suffix immediately follows row i's suffix
    // in t (the standard LF/next mapping): the j-th (0-indexed) occurrence
    // of value `symbols[i]` in the ORIGINAL (unsorted) array corresponds
    // to the (C[symbols[i]] + j)-th row in sorted order, and that row's
    // suffix is exactly "row i's suffix, extended by one character" --
    // i.e. what comes right after row i when reading t forwards.
    std::vector<int> next(m);
    std::array<int, kAlphabet> seen{};
    int sentinel_row = -1;
    for (int i = 0; i < m; i++) {
        u16 s = symbols[i];
        next[C[s] + seen[s]] = i;
        seen[s]++;
        if (s == 0) sentinel_row = i;
    }

    // Walk `next` starting from the sentinel's row (the row whose suffix
    // is the whole string t, i.e. position 0 in the original t) for m-1
    // steps, recovering t[0], t[1], ..., t[n-1] in order (t[n], the
    // sentinel itself, is never emitted -- the walk stops one step short
    // of returning to it).
    std::vector<u8> out;
    out.reserve(m - 1);
    int row = sentinel_row;
    for (int i = 0; i < m - 1; i++) {
        row = next[row];
        out.push_back((u8)(symbols[row] - 1));
    }
    return out;
}

} // namespace csa
