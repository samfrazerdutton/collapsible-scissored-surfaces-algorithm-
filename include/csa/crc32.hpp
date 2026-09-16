// csa/crc32.hpp — standard CRC-32 (IEEE 802.3 polynomial, the same one
// zlib/gzip/Ethernet/PNG use), table-based. A small, self-contained
// implementation rather than a dependency on the miniz vendored for
// `scissorc benchmark`'s gzip comparison -- that dependency is
// deliberately scoped to the CLI's optional benchmark command, not
// linked into csa_core, and packet integrity checking (packet_transport.hpp)
// is a core-library feature that needs to work in every build, including
// the WASM one.
#pragma once
#include "csa/common.hpp"
#include <cstddef>

namespace csa {

// The standard CRC-32/ISO-HDLC check value for the ASCII string
// "123456789" is 0xCBF43926 -- the universal test vector every real
// CRC-32 implementation is checked against; see test_crc32() in
// tests/test_main.cpp, not just this comment.
u32 crc32(const u8* data, size_t size);
inline u32 crc32(const std::vector<u8>& data) { return crc32(data.data(), data.size()); }

} // namespace csa
