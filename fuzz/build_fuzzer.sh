#!/bin/bash
# Builds fuzz/fuzz_decompress.cpp with clang + libFuzzer + ASan + UBSan.
# Not part of the CMake build (fuzzing is a manual/scheduled activity,
# not something every `cmake --build` should pay clang-specific-flag cost
# for) -- run this directly with clang installed (WSL Ubuntu's default
# clang worked for the run this project's own history records; see
# fuzz/README.md).
set -e
cd "$(dirname "$0")/.."
mkdir -p /tmp/fuzzobj

CXX_FUZZ="clang++ -std=c++17 -O1 -g -fsanitize=fuzzer,address,undefined -fno-omit-frame-pointer -Iinclude"
CXX_PLAIN="clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -Iinclude"

$CXX_FUZZ -c fuzz/fuzz_decompress.cpp -o /tmp/fuzzobj/fuzz_decompress.o

for f in range_coder pantograph_lift rod_joint_transform quaternion_joint pose_stream \
         rans_coder lz_matcher lz_codec bwt_transform bwt_codec codec simd pantograph_lift_cuda_stub \
         crc32 packet_transport; do
    $CXX_PLAIN -c "src/$f.cpp" -o "/tmp/fuzzobj/$f.o"
done

clang++ -std=c++17 -O1 -g -fsanitize=address,undefined -mavx2 -fno-omit-frame-pointer -Iinclude \
    -c src/simd_avx2.cpp -o /tmp/fuzzobj/simd_avx2.o

clang++ -fsanitize=fuzzer,address,undefined -o /tmp/fuzz_decompress /tmp/fuzzobj/*.o

echo "built /tmp/fuzz_decompress"
