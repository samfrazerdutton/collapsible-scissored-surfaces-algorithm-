#include "csa/bwt_codec.hpp"
#include "csa/bwt_transform.hpp"
#include "csa/range_coder.hpp"
#include <array>
#include <stdexcept>

namespace csa {

namespace {

constexpr int kMtfAlphabet = 257; // matches BwtBlockResult's 0=sentinel, 1..256=byte+1

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
        all_ranks[b] = mtf_encode(bwt.symbols);
    }

    std::vector<u8> coded;
    RangeEncoder enc(coded);
    FenwickFreqN model(kMtfAlphabet);
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

    FenwickFreqN model(kMtfAlphabet);
    std::vector<u8> result;
    result.reserve((size_t)original_length);
    for (u64 b = 0; b < num_blocks; b++) {
        size_t nsym = (size_t)block_lens[b] + 1; // +1 for the sentinel row
        std::vector<u16> ranks(nsym);
        for (size_t i = 0; i < nsym; i++) {
            u32 target = dec.get_freq(model.total());
            u32 lo, fr;
            int sym = model.find(target, lo, fr);
            dec.decode(lo, fr);
            model.update(sym);
            ranks[i] = (u16)sym;
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
