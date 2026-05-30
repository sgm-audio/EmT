add_executable(test_ssm_block test_ssm_block.cpp)
target_link_libraries(test_ssm_block PRIVATE core common GTest::gtest GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(test_ssm_block)

add_executable(test_weight_loader test_weight_loader.cpp)
target_link_libraries(test_weight_loader PRIVATE core common GTest::gtest GTest::gtest_main)
gtest_discover_tests(test_weight_loader)

add_executable(benchmark_ssm_block benchmark_ssm_block.cpp)
target_link_libraries(benchmark_ssm_block PRIVATE core common)
target_compile_options(benchmark_ssm_block PRIVATE ${RELEASE_COMPILE_OPTIONS})