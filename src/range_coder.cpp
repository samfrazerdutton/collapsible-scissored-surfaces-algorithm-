#include "csa/range_coder.hpp"

namespace csa {

std::vector<u8> range_encode_bytes(const std::vector<u8>& input) {
    std::vector<u8> out;
    out.reserve(input.size() / 2 + 16);
    RangeEncoder enc(out);
    Order1Model model;
    int ctx = 0;
    for (u8 byte : input) {
        FenwickFreq& f = model.context(ctx);
        u32 lo = f.cumfreq(byte);
        u32 fr = f.freq_of(byte);
        enc.encode(lo, fr, f.total());
        f.update(byte);
        ctx = byte;
    }
    enc.finish();
    return out;
}

std::vector<u8> range_decode_bytes(const u8* data, size_t size, size_t output_length) {
    std::vector<u8> out;
    out.reserve(output_length);
    RangeDecoder dec(data, size);
    Order1Model model;
    int ctx = 0;
    for (size_t i = 0; i < output_length; i++) {
        FenwickFreq& f = model.context(ctx);
        u32 target = dec.get_freq(f.total());
        u32 lo, fr;
        int sym = f.find(target, lo, fr);
        dec.decode(lo, fr);
        f.update(sym);
        out.push_back((u8)sym);
        ctx = sym;
    }
    return out;
}

} // namespace csa
