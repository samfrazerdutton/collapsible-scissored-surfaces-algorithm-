cmake_minimum_required(VERSION 3.18)

option(WITH_CUDA "Build the CUDA-accelerated Pantograph Lift kernel" ON)

if(WITH_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        project(collapsible_scissored_surfaces LANGUAGES CXX CUDA)
        set(CSA_HAVE_CUDA ON)
    else()
        project(collapsible_scissored_surfaces LANGUAGES CXX)
        set(CSA_HAVE_CUDA OFF)
        message(WARNING "WITH_CUDA was requested but no CUDA compiler was found; building CPU-only.")
    endif()
else()
    project(collapsible_scissored_surfaces LANGUAGES CXX)
    set(CSA_HAVE_CUDA OFF)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# csa_core is a static library linked into both scissorc (a plain
# executable, doesn't care) and csa_capi (a *shared* library, which does
# care): on Linux/GCC, a static lib built without -fPIC fails to link
# into a .so with "relocation ... can not be used when making a shared
# object" the moment it contains anything using the initial-exec TLS
# model, which src/codec.cpp's `static thread_local CudaLiftSession
# session` does. Harmless on MSVC (no such PIC/non-PIC distinction there),
# so this is set unconditionally rather than gated per-platform. Found by
# reproducing the CI Linux build locally (WSL) after ci.yml's first real
# run failed here -- not something the Windows/MSVC development this
# project used throughout would ever surface on its own.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>" "$<$<COMPILE_LANGUAGE:CXX>:/permissive->")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>" "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>" "$<$<COMPILE_LANGUAGE:CXX>:-O2>")
endif()

add_library(csa_core STATIC
    src/range_coder.cpp
    src/pantograph_lift.cpp
    src/rod_joint_transform.cpp
    src/quaternion_joint.cpp
    src/pose_stream.cpp
    src/rans_coder.cpp
    src/lz_matcher.cpp
    src/lz_codec.cpp
    src/bwt_transform.cpp
    src/bwt_codec.cpp
    src/codec.cpp
)
target_include_directories(csa_core PUBLIC include)

if(CSA_HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 20-series (Turing, sm_75); override with -DCMAKE_CUDA_ARCHITECTURES=... on other GPUs
    add_library(csa_cuda STATIC
        cuda/pantograph_lift_cuda.cu
        cuda/quaternion_calibration_cuda.cu
    )
    target_include_directories(csa_cuda PUBLIC include cuda)
    set_target_properties(csa_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(csa_core PUBLIC CSA_HAVE_CUDA=1)
    target_link_libraries(csa_core PUBLIC csa_cuda)

    # Standalone GPU-vs-CPU calibration benchmark (see DESIGN.md) -- not
    # part of the shipped library, only built when CUDA is available
    # since it directly measures the CUDA calibration kernel.
    add_executable(quat_calibration_gpu_crossover bench/quat_calibration_gpu_crossover.cpp)
    target_link_libraries(quat_calibration_gpu_crossover PRIVATE csa_core)
else()
    target_sources(csa_core PRIVATE src/pantograph_lift_cuda_stub.cpp)
    target_compile_definitions(csa_core PUBLIC CSA_HAVE_CUDA=0)
endif()

# miniz: a single-file (miniz.c/miniz.h), public-domain, zlib-compatible
# deflate/inflate implementation -- vendored (not a system dependency) so
# `scissorc benchmark` can compare CSA against a real gzip-equivalent
# baseline without relying on zlib being present on the build machine
# (it isn't, by default, on plain Windows). Populated without running
# miniz's own CMakeLists.txt (which calls project(miniz C) and would
# otherwise fight this project's own C++-only language/flags setup) --
# only its two source files are added directly to the scissorc target
# below. This is the one real new external dependency added for the
# "csa benchmark" feature; deliberately does not also vendor a real LZMA
# implementation (a much larger, multi-file dependency), so `scissorc
# benchmark`'s comparison is against gzip-equivalent only, not lzma --
# see its --explain output and BENCHMARKS.md for that disclosed scope.
include(FetchContent)
enable_language(C)
FetchContent_Declare(
    miniz
    GIT_REPOSITORY https://github.com/richgel999/miniz.git
    GIT_TAG 3.1.2
)
FetchContent_GetProperties(miniz)
if(NOT miniz_POPULATED)
    FetchContent_Populate(miniz)
endif()

# miniz.h unconditionally #includes "miniz_export.h", normally produced by
# its own CMakeLists.txt's generate_export_header() call (which we're
# deliberately not running -- see the comment above). Since miniz.c is
# compiled directly into scissorc as a static object, not a separate
# shared library, the export macro just needs to be empty; write that by
# hand rather than pull in miniz's own build system for one generated file.
file(WRITE "${miniz_SOURCE_DIR}/miniz_export.h" "#ifndef MINIZ_EXPORT\n#define MINIZ_EXPORT\n#endif\n")

# miniz.c alone (the zlib-compatible mz_compress2()/mz_uncompress()
# wrapper layer) isn't actually self-contained in this repo's git tree --
# it calls tdefl_*/tinfl_* functions whose real implementations live in
# the separate miniz_tdef.c/miniz_tinfl.c files (a true single-file
# amalgamation is only produced by amalgamate.sh for packaged releases,
# not committed as-is); found this the direct way, via a real unresolved-
# symbol link error, not by reading miniz's docs closely enough up front.
add_executable(scissorc cli/main.cpp
    "${miniz_SOURCE_DIR}/miniz.c"
    "${miniz_SOURCE_DIR}/miniz_tdef.c"
    "${miniz_SOURCE_DIR}/miniz_tinfl.c"
)
target_include_directories(scissorc PRIVATE "${miniz_SOURCE_DIR}")
target_link_libraries(scissorc PRIVATE csa_core)

# libcsa: stable C ABI shared library for language bindings (Python, Rust,
# C#, Go, ...) -- see include/csa/csa_capi.h. No C++ types cross this
# boundary, so it links the same way regardless of which toolchain built
# it vs. which one is loading it.
add_library(csa_capi SHARED src/csa_capi.cpp)
target_link_libraries(csa_capi PRIVATE csa_core)
target_include_directories(csa_capi PUBLIC include)
target_compile_definitions(csa_capi PRIVATE CSA_BUILDING_DLL)
set_target_properties(csa_capi PROPERTIES
    WINDOWS_EXPORT_ALL_SYMBOLS ON   # belt-and-braces alongside the __declspec macros in csa_capi.h
    OUTPUT_NAME "csa"                # -> csa.dll / libcsa.so / libcsa.dylib
)

# `pip install .` (pyproject.toml, scikit-build-core) installs just this
# one target, placing the compiled shared library directly alongside
# bindings/python/csa.py -- see that file's _find_library() for the
# matching runtime lookup. RUNTIME is where Windows puts a shared
# library's actual .dll (its "library" file in CMake's model is the
# small .lib import stub, not the .dll); LIBRARY/ARCHIVE cover
# .so/.dylib and static-linkage cases respectively so this install rule
# is correct across platforms without conditionals.
install(TARGETS csa_capi
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
    ARCHIVE DESTINATION .
)

enable_testing()
add_executable(csa_tests tests/test_main.cpp)
target_link_libraries(csa_tests PRIVATE csa_core)
add_test(NAME csa_tests COMMAND csa_tests)

# Links against the actual shared library (not csa_core directly) so a
# real symbol-export/linking problem in the C ABI can't hide behind the
# C++ tests, which never cross that boundary.
add_executable(csa_capi_test tests/test_capi.cpp)
target_link_libraries(csa_capi_test PRIVATE csa_capi)
add_test(NAME csa_capi_test COMMAND csa_capi_test)
