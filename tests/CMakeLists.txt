# SSMWeights is ~4 MB at max config — MSVC's 1 MB default stack is too small.
if(MSVC)
    set(_MAMBA_TEST_STACK /STACK:16777216)
endif()

add_executable(test_ssm_block test_ssm_block.cpp)
target_link_libraries(test_ssm_block PRIVATE core common GTest::gtest GTest::gtest_main)
if(MSVC)
    target_link_options(test_ssm_block PRIVATE "/STACK:16777216")
endif()
include(GoogleTest)
gtest_discover_tests(test_ssm_block)

# Needed for std::thread in test_conditioning.cpp
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_executable(test_weight_loader test_weight_loader.cpp)
target_link_libraries(test_weight_loader PRIVATE core common GTest::gtest GTest::gtest_main)
if(MSVC)
    target_link_options(test_weight_loader PRIVATE "/STACK:16777216")
endif()
gtest_discover_tests(test_weight_loader)

add_executable(test_conditioning test_conditioning.cpp)
target_link_libraries(test_conditioning PRIVATE core common GTest::gtest GTest::gtest_main Threads::Threads)
if(MSVC)
    target_link_options(test_conditioning PRIVATE "/STACK:16777216")
endif()
gtest_discover_tests(test_conditioning)

# MSVC uses /O2 for Release; on other compilers we already have -O3 via
# CMAKE_CXX_FLAGS_RELEASE.  The benchmark wants the most aggressive
# optimization the compiler can offer, so add an extra one if available.
if(MSVC)
    set(_MAMBA_BENCH_OPT_FLAGS /O2)
else()
    set(_MAMBA_BENCH_OPT_FLAGS -O3)
endif()

add_executable(benchmark_ssm_block benchmark_ssm_block.cpp)
target_link_libraries(benchmark_ssm_block PRIVATE core common)
target_compile_options(benchmark_ssm_block PRIVATE ${_MAMBA_BENCH_OPT_FLAGS})
if(MSVC)
    target_link_options(benchmark_ssm_block PRIVATE "/STACK:16777216")
endif()

add_executable(test_stability test_stability.cpp)
target_link_libraries(test_stability PRIVATE core common GTest::gtest GTest::gtest_main)
if(MSVC)
    target_link_options(test_stability PRIVATE "/STACK:16777216")
endif()
gtest_discover_tests(test_stability)

add_executable(test_realtime_safety test_realtime_safety.cpp)
target_link_libraries(test_realtime_safety PRIVATE core common GTest::gtest GTest::gtest_main)
if(MSVC)
    target_link_options(test_realtime_safety PRIVATE "/STACK:16777216")
endif()
gtest_discover_tests(test_realtime_safety)

add_executable(test_cpp_parity test_cpp_parity.cpp)
target_link_libraries(test_cpp_parity PRIVATE core common GTest::gtest GTest::gtest_main)
target_compile_definitions(test_cpp_parity PRIVATE
    MAMBA_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
if(MSVC)
    target_link_options(test_cpp_parity PRIVATE "/STACK:16777216")
endif()
gtest_discover_tests(test_cpp_parity)
