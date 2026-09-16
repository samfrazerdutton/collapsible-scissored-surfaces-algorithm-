#include "csa/crc32.hpp"

namespace csa {

namespace {

// Standard reflected CRC-32 table (polynomial 0xEDB88320), built once at
// static-init time rather than checked in as a 256-entry literal -- the
// generation loop is the same one every reference implementation uses,
// so building it is exactly as trustworthy as copying it, without 256
// lines of magic numbers no reader can independently verify.
struct Crc32Table {
    u32 t[256];
    Crc32Table() {
        for (u32 i = 0; i < 256; i++) {
            u32 c = i;
            for (int k = 0; k < 8; k++)
                c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
            t[i] = c;
        }
    }
};

const Crc32Table& table() {
    static const Crc32Table instance;
    return instance;
}

} // namespace

u32 crc32(const u8* data, size_t size) {
    const u32* t = table().t;
    u32 c = 0xFFFFFFFFu;
    for (size_t i = 0; i < size; i++)
        c = t[(c ^ data[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

} // namespace csa
