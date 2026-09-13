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

if(MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/W4>" "$<$<COMPILE_LANGUAGE:CXX>:/permissive->")
else()
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall>" "$<$<COMPILE_LANGUAGE:CXX>:-Wextra>" "$<$<COMPILE_LANGUAGE:CXX>:-O2>")
endif()

add_library(csa_core STATIC
    src/range_coder.cpp
    src/pantograph_lift.cpp
    src/rod_joint_transform.cpp
    src/codec.cpp
)
target_include_directories(csa_core PUBLIC include)

if(CSA_HAVE_CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 75) # RTX 20-series (Turing, sm_75); override with -DCMAKE_CUDA_ARCHITECTURES=... on other GPUs
    add_library(csa_cuda STATIC
        cuda/pantograph_lift_cuda.cu
    )
    target_include_directories(csa_cuda PUBLIC include cuda)
    set_target_properties(csa_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_definitions(csa_core PUBLIC CSA_HAVE_CUDA=1)
    target_link_libraries(csa_core PUBLIC csa_cuda)
else()
    target_sources(csa_core PRIVATE src/pantograph_lift_cuda_stub.cpp)
    target_compile_definitions(csa_core PUBLIC CSA_HAVE_CUDA=0)
endif()

add_executable(scissorc cli/main.cpp)
target_link_libraries(scissorc PRIVATE csa_core)

enable_testing()
add_executable(csa_tests tests/test_main.cpp)
target_link_libraries(csa_tests PRIVATE csa_core)
add_test(NAME csa_tests COMMAND csa_tests)
