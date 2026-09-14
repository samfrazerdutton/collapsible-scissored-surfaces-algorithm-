#include "csa/bwt_codec.hpp"
#include "csa/bwt_transform.hpp"
#include "csa/range_coder.hpp"
#include <array>
#include <stdexcept>

namespace csa {

namespace {

constexpr int kMtfAlphabet = 257; // matches BwtBlockResult's 0=sentinel, 1..256=byte+1
// After the zero-run substitution below, the alphabet gains two reserved
// symbols (RUNA, RUNB) and every nonzero MTF rank shifts up by one to
// make room for them: RUNA=0, RUNB=1, shifted nonzero ranks 2..257.
constexpr int kRleAlphabet = kMtfAlphabet + 1;

// Move-to-front: turns BWT's locally-clustered symbol runs into mostly-
// small "how far back in recent history was this value last seen"
// numbers, which the entropy coder that follows can then exploit.
std::vector<u16> mtf_encode(const std::vector<u16>& symbols) {
    std::array<u16, kMtfAlphabet> table{};
    for (int i = 0; i < kMtfAlphabet; i++) table[i] = (u16)i;

    std::vector<u16> out(symbols.size());
    for (size_t i = 0; i < symbols.size(); i++) {
        u16 s = symbols[i];
        int pos = 0;
        while (table[pos] != s) pos++;
        out[i] = (u16)pos;
        for (int k = pos; k > 0; k--) table[k] = table[k - 1];
        table[0] = s;
    }
    return out;
}

std::vector<u16> mtf_decode(const std::vector<u16>& ranks) {
    std::array<u16, kMtfAlphabet> table{};
    for (int i = 0; i < kMtfAlphabet; i++) table[i] = (u16)i;

    std::vector<u16> out(ranks.size());
    for (size_t i = 0; i < ranks.size(); i++) {
        int pos = ranks[i];
        u16 s = table[pos];
        out[i] = s;
        for (int k = pos; k > 0; k--) table[k] = table[k - 1];
        table[0] = s;
    }
    return out;
}

// bzip2's RUNA/RUNB zero-run encoding: MTF's rank-0 symbol dominates its
// output on BWT-clustered data (every byte that matches the most-
// recently-seen one produces rank 0), and long runs of it are exactly
// the pattern this substitution targets -- encoding a run of N zeros as
// O(log N) reserved-symbol tokens instead of N individual rank-0 symbols
// through the entropy coder. A run length N (N >= 1) is represented in
// bijective base-2 (digit set {1, 2}, least-significant digit first):
// repeatedly take d = ((N-1) mod 2) + 1, emit RUNA for d=1 or RUNB for
// d=2, then N = (N-d)/2, until N == 0. This is a bijection (every N >= 1
// has exactly one such digit sequence), so decode is unambiguous: given
// digits d_0, d_1, ..., N = sum(d_i * 2^i).
constexpr u16 kRunA = 0;
constexpr u16 kRunB = 1;

std::vector<u16> rle_encode_zero_runs(const std::vector<u16>& ranks) {
    std::vector<u16> out;
    out.reserve(ranks.size());
    size_t i = 0;
    while (i < ranks.size()) {
        if (ranks[i] == 0) {
            size_t run = 0;
            while (i < ranks.size() && ranks[i] == 0) { run++; i++; }
            u64 n = run;
            while (n > 0) {
                u64 d = ((n - 1) % 2) + 1;
                out.push_back(d == 1 ? kRunA : kRunB);
                n = (n - d) / 2;
            }
        } else {
            out.push_back((u16)(ranks[i] + 1)); // shift nonzero ranks up by one
            i++;
        }
    }
    return out;
}

// `expected_len` is the exact number of MTF ranks this block must
// decode to (known up front from the block's stored length -- see
// bwt_encode/bwt_decode), which is what lets this correctly flush a
// zero-run that runs all the way to the end of the block.
std::vector<u16> rle_decode_zero_runs(const std::vector<u16>& rle_symbols, size_t expected_len) {
    std::vector<u16> out;
    out.reserve(expected_len);
    size_t i = 0;
    while (out.size() < expected_len) {
        u64 n = 0, pow2 = 1;
        bool have_digits = false;
        while (i < rle_symbols.size() && (rle_symbols[i] == kRunA || rle_symbols[i] == kRunB)) {
            u64 d = (rle_symbols[i] == kRunA) ? 1 : 2;
            n += d * pow2;
            pow2 *= 2;
            have_digits = true;
            i++;
        }
        if (have_digits) {
            for (u64 k = 0; k < n; k++) out.push_back(0);
        }
        if (out.size() >= expected_len) break;
        out.push_back((u16)(rle_symbols[i] - 1)); // undo the nonzero-rank shift
        i++;
    }
    return out;
}

} // namespace

std::vector<u8> bwt_encode(const std::vector<u8>& input) {
    std::vector<u8> out;
    put_u64(out, (u64)input.size());
    if (input.empty()) return out;

    size_t block_size = kBwtDefaultBlockSize;
    size_t num_blocks = (input.size() + block_size - 1) / block_size;
    put_u64(out, (u64)num_blocks);

    std::vector<std::vector<u16>> all_ranks(num_blocks);
    for (size_t b = 0; b < num_blocks; b++) {
        size_t start = b * block_size;
        size_t end = std::min(input.size(), start + block_size);
        std::vector<u8> block(input.begin() + (ptrdiff_t)start, input.begin() + (ptrdiff_t)end);
        put_u64(out, (u64)block.size());

        BwtBlockResult bwt = bwt_encode_block(block);
        std::vector<u16> ranks = mtf_encode(bwt.symbols);
        all_ranks[b] = rle_encode_zero_runs(ranks);
    }

    std::vector<u8> coded;
    RangeEncoder enc(coded);
    FenwickFreqN model(kRleAlphabet);
    for (auto& ranks : all_ranks) {
        for (u16 v : ranks) {
            u32 lo = model.cumfreq(v), fr = model.freq_of(v);
            enc.encode(lo, fr, model.total());
            model.update(v);
        }
    }
    enc.finish();

    put_u64(out, (u64)coded.size());
    out.insert(out.end(), coded.begin(), coded.end());
    return out;
}

std::vector<u8> bwt_decode(const u8* data, size_t size, size_t& pos) {
    u64 original_length = get_u64(data, size, pos);
    if (original_length == 0) return {};

    u64 num_blocks = get_u64(data, size, pos);
    std::vector<u64> block_lens(num_blocks);
    for (u64 b = 0; b < num_blocks; b++) block_lens[b] = get_u64(data, size, pos);

    u64 coded_len = get_u64(data, size, pos);
    if (pos + coded_len > size) throw std::runtime_error("csa: truncated bwt coded stream");
    RangeDecoder dec(data + pos, (size_t)coded_len);
    pos += (size_t)coded_len;

    FenwickFreqN model(kRleAlphabet);
    std::vector<u8> result;
    result.reserve((size_t)original_length);
    for (u64 b = 0; b < num_blocks; b++) {
        size_t nsym = (size_t)block_lens[b] + 1; // +1 for the sentinel row

        // The RLE stream has fewer symbols than `nsym` whenever a zero-run
        // got substituted, so this can't just loop nsym times over the
        // entropy coder -- it decodes one RLE symbol at a time, expanding
        // runs as they're recognized, until exactly nsym ranks have been
        // produced (a run is allowed to end precisely at the block
        // boundary, with no real symbol following it in this block; the
        // early check below handles that without over-reading into the
        // next block's symbols).
        std::vector<u16> ranks;
        ranks.reserve(nsym);
        u64 pending_n = 0, pending_pow2 = 1;
        bool have_pending = false;
        auto flush_pending = [&]() {
            if (have_pending) {
                for (u64 k = 0; k < pending_n; k++) ranks.push_back(0);
                pending_n = 0;
                pending_pow2 = 1;
                have_pending = false;
            }
        };
        while (ranks.size() < nsym) {
            if (have_pending && ranks.size() + pending_n >= nsym) {
                flush_pending();
                break;
            }
            u32 target = dec.get_freq(model.total());
            u32 lo, fr;
            int sym = model.find(target, lo, fr);
            dec.decode(lo, fr);
            model.update(sym);

            if (sym == kRunA || sym == kRunB) {
                u64 d = (sym == kRunA) ? 1 : 2;
                pending_n += d * pending_pow2;
                pending_pow2 *= 2;
                have_pending = true;
            } else {
                flush_pending();
                ranks.push_back((u16)(sym - 1));
            }
        }

        std::vector<u16> symbols = mtf_decode(ranks);
        std::vector<u8> block = bwt_decode_block(symbols);
        result.insert(result.end(), block.begin(), block.end());
    }

    if (result.size() != original_length)
        throw std::runtime_error("csa: bwt decode length mismatch");
    return result;
}

} // namespace csa
