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

// SA-IS: Nong/Zhang/Chen's linear-time suffix array construction via
// induced sorting. `t`'s last element must be a unique value strictly
// smaller than every other element (the same sentinel convention
// bwt_encode_block already uses), and `alphabet_size` is one past the
// largest value that can appear in `t` (valid symbols are
// 0..alphabet_size-1).
//
// Kept alongside the prefix-doubling `build_suffix_array` above rather
// than replacing it outright: correctness for an algorithm this
// intricate is trusted through exhaustive cross-validation against that
// already "banana"-verified implementation (see
// test_bwt_transform_sais_matches_reference), not just a handful of
// hand-checked examples -- the same standard this project applies to
// every other genuinely tricky piece of math in it.
std::vector<int> build_suffix_array_sais(const std::vector<int>& t, int alphabet_size) {
    int n = (int)t.size();
    if (n == 0) return {};
    if (n == 1) return {0};

    // --- Classify every suffix S-type or L-type. The sentinel position
    // is S-type by definition; comparing right-to-left from there, a
    // suffix is S-type if it's lexicographically smaller than the one
    // immediately after it (or equal in first character and the next
    // suffix is itself S-type). ---
    std::vector<u8> is_s(n);
    is_s[n - 1] = 1;
    for (int i = n - 2; i >= 0; i--) {
        if (t[i] < t[i + 1]) is_s[i] = 1;
        else if (t[i] > t[i + 1]) is_s[i] = 0;
        else is_s[i] = is_s[i + 1];
    }
    // An "LMS" (leftmost S-type) position is S-type with an L-type
    // predecessor -- position 0 can never be LMS (no predecessor).
    // Given the sentinel is always S-type and the position before it
    // always holds a strictly larger real value (hence L-type), the
    // sentinel's own position is always classified LMS too, with no
    // special-casing needed anywhere below.
    auto is_lms = [&](int i) { return i > 0 && is_s[i] != 0 && is_s[i - 1] == 0; };

    std::vector<int> bucket_count(alphabet_size, 0);
    for (int c : t) bucket_count[c]++;
    auto bucket_starts = [&]() {
        std::vector<int> starts(alphabet_size);
        int sum = 0;
        for (int c = 0; c < alphabet_size; c++) { starts[c] = sum; sum += bucket_count[c]; }
        return starts;
    };
    auto bucket_ends = [&]() {
        std::vector<int> ends(alphabet_size);
        int sum = 0;
        for (int c = 0; c < alphabet_size; c++) { sum += bucket_count[c]; ends[c] = sum; }
        return ends;
    };

    // Places the given LMS positions at the tails of their character
    // buckets (in the order provided, processed back-to-front so earlier
    // entries end up earlier within a shared bucket), then induces every
    // L-type suffix left-to-right and every S-type suffix right-to-left.
    // Called twice: once with an arbitrary (appearance-order) placement
    // to discover the LMS suffixes' true relative order, and once more
    // with that now-correct order to produce the final suffix array.
    auto induced_sort = [&](const std::vector<int>& lms_order) {
        std::vector<int> sa(n, -1);
        std::vector<int> tail = bucket_ends();
        for (int k = (int)lms_order.size() - 1; k >= 0; k--) {
            int i = lms_order[k];
            int c = t[i];
            tail[c]--;
            sa[tail[c]] = i;
        }

        std::vector<int> head = bucket_starts();
        for (int i = 0; i < n; i++) {
            if (sa[i] <= 0) continue;
            int j = sa[i] - 1;
            if (!is_s[j]) {
                int c = t[j];
                sa[head[c]] = j;
                head[c]++;
            }
        }

        tail = bucket_ends();
        for (int i = n - 1; i >= 0; i--) {
            if (sa[i] <= 0) continue;
            int j = sa[i] - 1;
            if (is_s[j]) {
                int c = t[j];
                tail[c]--;
                sa[tail[c]] = j;
            }
        }
        return sa;
    };

    std::vector<int> lms_appearance_order;
    for (int i = 0; i < n; i++)
        if (is_lms(i)) lms_appearance_order.push_back(i);

    std::vector<int> sa1 = induced_sort(lms_appearance_order);

    std::vector<int> lms_sorted_by_sa1;
    lms_sorted_by_sa1.reserve(lms_appearance_order.size());
    for (int i = 0; i < n; i++)
        if (sa1[i] >= 0 && is_lms(sa1[i])) lms_sorted_by_sa1.push_back(sa1[i]);

    // Two LMS substrings (position to next LMS position, inclusive) are
    // equal only if they have the same length and identical characters
    // throughout. The sentinel's own LMS position (length-1 substring,
    // unique value) can never equal any other, so it's handled by the
    // same general loop with no special case.
    auto lms_substrings_equal = [&](int p, int q) {
        for (int k = 0;; k++) {
            bool p_end = k > 0 && is_lms(p + k);
            bool q_end = k > 0 && is_lms(q + k);
            if (p_end && q_end) return true;
            if (p_end != q_end) return false;
            if (t[p + k] != t[q + k]) return false;
        }
    };

    std::vector<int> name(n, -1);
    int num_names = 1;
    name[lms_sorted_by_sa1[0]] = 0;
    for (size_t k = 1; k < lms_sorted_by_sa1.size(); k++) {
        int prev = lms_sorted_by_sa1[k - 1];
        int curr = lms_sorted_by_sa1[k];
        if (!lms_substrings_equal(prev, curr)) num_names++;
        name[curr] = num_names - 1;
    }

    std::vector<int> lms_final_order;
    if (num_names == (int)lms_appearance_order.size()) {
        // Every LMS substring is distinct -- the order already extracted
        // from the first induced sort is already the correct final
        // relative order of the LMS suffixes, no recursion needed.
        lms_final_order = lms_sorted_by_sa1;
    } else {
        // Reduce to a string of names (in original LMS appearance order)
        // and recurse. This reduced string's own last character is
        // always name 0 -- the original sentinel's LMS position is
        // always the last one in appearance order, and is always named
        // 0 (nothing sorts before it) -- exactly the unique-smallest-at-
        // the-end property this function requires, so no additional
        // sentinel needs to be appended for the recursive call.
        int reduced_n = (int)lms_appearance_order.size();
        std::vector<int> reduced_string(reduced_n);
        for (int k = 0; k < reduced_n; k++) reduced_string[k] = name[lms_appearance_order[k]];

        std::vector<int> reduced_sa = build_suffix_array_sais(reduced_string, num_names);

        lms_final_order.resize(reduced_n);
        for (int k = 0; k < reduced_n; k++) lms_final_order[k] = lms_appearance_order[reduced_sa[k]];
    }

    return induced_sort(lms_final_order);
}

} // namespace

std::vector<int> bwt_debug_build_suffix_array_reference(const std::vector<int>& t) {
    return build_suffix_array(t);
}

std::vector<int> bwt_debug_build_suffix_array_sais(const std::vector<int>& t, int alphabet_size) {
    return build_suffix_array_sais(t, alphabet_size);
}

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

    std::vector<int> sa = build_suffix_array_sais(t, 257);

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
