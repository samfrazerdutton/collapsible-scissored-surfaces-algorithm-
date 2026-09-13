#include "csa/lz_codec.hpp"
#include "csa/bitpacker.hpp"
#include "csa/lz_matcher.hpp"
#include "csa/range_coder.hpp"
#include <stdexcept>

namespace csa {

namespace {

constexpr int kLitAlphabet = 257; // 0-255 literal byte, 256 = match flag
constexpr int kMatchFlag = 256;
constexpr int kNumBuckets = 48; // headroom well past any realistic file size

// Magnitude-class ("bucket") decomposition of a non-negative integer:
// bucket = floor(log2(v+1)); the low `bucket` bits of (v+1) beyond its
// leading 1 are the "extra bits", packed raw since they're ~uniform
// within a bucket by construction. Both match length (offset by
// -kLzMinMatch) and distance (offset by -1) are encoded via this same
// scheme -- see the header comment for why that's the same shape.
void bucket_encode(u32 v, int& bucket, int& extra_bits, u32& extra_value) {
    u64 x = (u64)v + 1;
    int b = 0;
    while ((u64(1) << (b + 1)) <= x) b++;
    bucket = b;
    extra_bits = b;
    extra_value = (u32)(x - (u64(1) << b));
}

u32 bucket_decode(int bucket, u32 extra_value) {
    u64 x = (u64(1) << bucket) + extra_value;
    return (u32)(x - 1);
}

} // namespace

std::vector<u8> lz_encode(const std::vector<u8>& input, int max_chain, size_t nice_length) {
    std::vector<u8> out;
    put_u64(out, (u64)input.size());

    if (input.empty()) {
        put_u64(out, 0); // coded_length
        put_u64(out, 0); // extra_bits_length
        return out;
    }

    std::vector<LzToken> tokens = lz_parse(input, max_chain, nice_length);

    std::vector<u8> coded;
    RangeEncoder enc(coded);
    Order1ModelN lit_model(kLitAlphabet);
    FenwickFreqN len_model(kNumBuckets);
    FenwickFreqN dist_model(kNumBuckets);
    std::vector<u8> extra;
    BitWriter bw(extra);

    size_t pos = 0;
    int ctx = 0;
    for (const auto& t : tokens) {
        if (!t.is_match) {
            FenwickFreqN& m = lit_model.context(ctx);
            u32 lo = m.cumfreq(t.literal), fr = m.freq_of(t.literal);
            enc.encode(lo, fr, m.total());
            m.update(t.literal);
            ctx = t.literal;
            pos += 1;
        } else {
            FenwickFreqN& m = lit_model.context(ctx);
            u32 lo = m.cumfreq(kMatchFlag), fr = m.freq_of(kMatchFlag);
            enc.encode(lo, fr, m.total());
            m.update(kMatchFlag);

            int lb, lbits;
            u32 lex;
            bucket_encode(t.length - kLzMinMatch, lb, lbits, lex);
            u32 llo = len_model.cumfreq(lb), lfr = len_model.freq_of(lb);
            enc.encode(llo, lfr, len_model.total());
            len_model.update(lb);
            bw.write_bits(lex, lbits);

            int db, dbits;
            u32 dex;
            bucket_encode(t.distance - 1, db, dbits, dex);
            u32 dlo = dist_model.cumfreq(db), dfr = dist_model.freq_of(db);
            enc.encode(dlo, dfr, dist_model.total());
            dist_model.update(db);
            bw.write_bits(dex, dbits);

            pos += t.length;
            ctx = input[pos - 1];
        }
    }
    bw.flush();
    enc.finish();

    put_u64(out, (u64)coded.size());
    out.insert(out.end(), coded.begin(), coded.end());
    put_u64(out, (u64)extra.size());
    out.insert(out.end(), extra.begin(), extra.end());
    return out;
}

std::vector<u8> lz_decode(const u8* data, size_t size, size_t& pos) {
    u64 original_length = get_u64(data, size, pos);
    u64 coded_length = get_u64(data, size, pos);
    if (pos + coded_length > size) throw std::runtime_error("csa: truncated lz coded stream");
    const u8* coded_ptr = data + pos;
    pos += (size_t)coded_length;

    u64 extra_length = get_u64(data, size, pos);
    if (pos + extra_length > size) throw std::runtime_error("csa: truncated lz extra-bits stream");
    const u8* extra_ptr = data + pos;
    pos += (size_t)extra_length;

    std::vector<u8> result;
    if (original_length == 0) return result;
    result.reserve((size_t)original_length);

    RangeDecoder dec(coded_ptr, (size_t)coded_length);
    BitReader br(extra_ptr, (size_t)extra_length);
    Order1ModelN lit_model(kLitAlphabet);
    FenwickFreqN len_model(kNumBuckets);
    FenwickFreqN dist_model(kNumBuckets);

    int ctx = 0;
    while (result.size() < original_length) {
        FenwickFreqN& m = lit_model.context(ctx);
        u32 target = dec.get_freq(m.total());
        u32 lo, fr;
        int sym = m.find(target, lo, fr);
        dec.decode(lo, fr);
        m.update(sym);

        if (sym < 256) {
            result.push_back((u8)sym);
            ctx = sym;
        } else {
            u32 ltarget = dec.get_freq(len_model.total());
            u32 llo, lfr;
            int lb = len_model.find(ltarget, llo, lfr);
            dec.decode(llo, lfr);
            len_model.update(lb);
            u32 lex = (u32)br.read_bits(lb);
            u32 length = bucket_decode(lb, lex) + kLzMinMatch;

            u32 dtarget = dec.get_freq(dist_model.total());
            u32 dlo, dfr;
            int db = dist_model.find(dtarget, dlo, dfr);
            dec.decode(dlo, dfr);
            dist_model.update(db);
            u32 dex = (u32)br.read_bits(db);
            u32 distance = bucket_decode(db, dex) + 1;

            if (distance > result.size()) throw std::runtime_error("csa: lz match distance out of range");
            size_t start = result.size() - distance;
            for (u32 k = 0; k < length; k++) result.push_back(result[start + k]);
            ctx = result.back();
        }
    }
    return result;
}

} // namespace csa
