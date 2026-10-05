# Shared test helpers (tests/fixtures/*). Header-only, no GTest dependency.
add_library(test_fixtures INTERFACE)
target_include_directories(test_fixtures INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

include(GoogleTest)

# Needed for std::thread in test_conditioning.cpp and test_ara.cpp.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Test executables get warnings but keep C++ exceptions/RTTI: GTest needs
# exceptions, and several fixtures use throwing APIs (std::filesystem).
# MSVC targets get a 16 MB stack: SSMWeights is ~4 MB at max config while
# the default 1 MB stack is too small.
function(emt_add_test target)
    add_executable(${target} ${ARGN})
    target_link_libraries(${target} PRIVATE
        core common test_fixtures GTest::gtest GTest::gtest_main)
    target_compile_options(${target} PRIVATE ${EMT_WARN_FLAGS})
    if(MSVC)
        target_link_options(${target} PRIVATE "/STACK:16777216")
    endif()
    gtest_discover_tests(${target})
endfunction()

emt_add_test(test_ssm_block test_ssm_block.cpp)

emt_add_test(test_weight_loader test_weight_loader.cpp)

emt_add_test(test_conditioning
    test_conditioning.cpp
    ${PROJECT_SOURCE_DIR}/src/clap/clap_presets.cpp)
target_link_libraries(test_conditioning PRIVATE Threads::Threads)
target_include_directories(test_conditioning PRIVATE
    ${PROJECT_SOURCE_DIR}/extern/clap/include)

# Optimization level for the benchmark translation unit.  The root
# CMakeLists already sets -O3 -mavx2 -mfma (or /O2 /arch:AVX2) in the Release
# flags, and target_compile_options appends *after* those - so this is a
# deliberate belt-and-braces repeat of the optimization level, not an
# override of the ISA flags (which are not optimization-level flags and
# cannot be cancelled by a later -O3).  The measured cost lives in `core`,
# which is compiled with the Release flags above.
if(MSVC)
    set(_MAMBA_BENCH_OPT_FLAGS /O2)
else()
    set(_MAMBA_BENCH_OPT_FLAGS -O3)
endif()

add_executable(benchmark_ssm_block benchmark_ssm_block.cpp)
target_link_libraries(benchmark_ssm_block PRIVATE core common)
target_compile_options(benchmark_ssm_block PRIVATE
    ${EMT_WARN_FLAGS} ${_MAMBA_BENCH_OPT_FLAGS})
if(MSVC)
    target_link_options(benchmark_ssm_block PRIVATE "/STACK:16777216")
endif()

emt_add_test(test_stability test_stability.cpp)

emt_add_test(test_realtime_safety
    test_realtime_safety.cpp
    ${PROJECT_SOURCE_DIR}/src/clap/clap_fx_plugin.cpp
    ${PROJECT_SOURCE_DIR}/src/clap/clap_audio_ports.cpp
    ${PROJECT_SOURCE_DIR}/src/clap/clap_latency.cpp)
target_include_directories(test_realtime_safety PRIVATE
    ${PROJECT_SOURCE_DIR}/extern/clap/include)

emt_add_test(test_cpp_parity test_cpp_parity.cpp)

emt_add_test(test_ara test_ara.cpp)
target_link_libraries(test_ara PRIVATE ara_plugin Threads::Threads)
