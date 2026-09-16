// A real libFuzzer harness (built with clang's -fsanitize=fuzzer,undefined,address
// -- see fuzz/README.md for the exact commands this was actually run
// with) against this codec's real untrusted-input boundary: the four
// decompress functions a caller (CLI, Python, WASM/browser) hands
// arbitrary bytes to. The contract under fuzzing is exactly what
// FORMAT.md and this project's own test suite already state: malformed
// input must throw a clean std::runtime_error (or std::exception more
// generally), never crash, never hang, never read out of bounds, never
// attempt an unbounded allocation from an attacker-controlled length
// field. This harness does not assert decompression *succeeds* on
// anything -- only that it fails safely when it fails.
//
// The first byte of each fuzzer-provided input selects which of the
// four real decompress functions gets the rest of the bytes -- one
// corpus covers all four related parsers instead of needing four
// separate fuzz binaries and corpora for what is fundamentally the same
// "handle adversarial bytes safely" property repeated four times.
#include "csa/codec.hpp"
#include <cstdint>
#include <exception>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 1) return 0;
    uint8_t selector = data[0];
    std::vector<csa::u8> bytes(data + 1, data + size);

    try {
        switch (selector % 4) {
            case 0: { auto r = csa::decompress(bytes); (void)r; break; }
            case 1: { auto r = csa::decompress_geo2d(bytes); (void)r; break; }
            case 2: { auto r = csa::decompress_geo3d(bytes); (void)r; break; }
            case 3: { auto r = csa::decompress_pose(bytes); (void)r; break; }
        }
    } catch (const std::exception&) {
        // Expected, frequent, and correct: malformed input is supposed to
        // land here, not crash the process. See FORMAT.md's "known gap:
        // no CRC" section -- this harness is exactly what stands in for
        // that missing checksum today: not detecting corruption, but
        // guaranteeing detected-or-not, the process survives it.
    }
    return 0;
}
