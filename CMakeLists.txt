cmake_minimum_required(VERSION 3.20)
project(EmT VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    set(EMT_WARN_FLAGS /W4 /permissive- /wd4324 /wd4996 /D_CRT_SECURE_NO_WARNINGS)
    set(EMT_RT_FLAGS /GR-)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
    set(EMT_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(EMT_RT_FLAGS -fno-exceptions -fno-rtti -fvisibility=hidden)
endif()

# EMT_RT_FLAGS (no exceptions/RTTI, hidden visibility) applies to the
# real-time plugin libraries ONLY. It is deliberately NOT global:
# third-party code (GTest) and several test fixtures (std::filesystem)
# require exceptions. Applied per-target below, after add_subdirectory.
# EMT_WARN_FLAGS applies to every first-party target (libs + tests).

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()

# GTest source: FetchContent by default; -DEMT_USE_SYSTEM_GTEST=ON uses a
# system install instead. Offline/vendored builds can also pass
# -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path> (with
# -DFETCHCONTENT_FULLY_DISCONNECTED=ON to skip the download attempt).
option(EMT_USE_SYSTEM_GTEST "Use system GTest instead of FetchContent" OFF)
if(EMT_USE_SYSTEM_GTEST)
    find_package(GTest REQUIRED)
else()
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
        URL_HASH SHA256=8ad598c73ad796e0d8280b082cebd82a630d73e73cd3c70057938a6501bba5d7
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    set(gtest_force_shared_crt ON)
    set(BUILD_GMOCK OFF)
    set(INSTALL_GTEST OFF)
    FetchContent_MakeAvailable(googletest)
endif()

# NOTE: no global include_directories(). Each library declares its own
# includes (core/common/ara_plugin) or takes them transitively via linking.
# The two test targets that compile CLAP sources add extern/clap/include
# privately in tests/CMakeLists.txt.

add_subdirectory(src/core)
add_subdirectory(src/common)
add_subdirectory(src/ara)
add_subdirectory(src/clap)

foreach(_tgt IN ITEMS core ara_plugin mamba_synth)
    target_compile_options(${_tgt} PRIVATE ${EMT_RT_FLAGS} ${EMT_WARN_FLAGS})
endforeach()

enable_testing()
add_subdirectory(tests)

# clap-validator: runs the CLAP plugin validator against the built bundle
# when the tool is installed; otherwise the target explains the skip.
find_program(CLAP_VALIDATOR_BIN clap-validator)
if(CLAP_VALIDATOR_BIN)
    add_test(NAME clap-validator
        COMMAND ${CLAP_VALIDATOR_BIN} validate "$<TARGET_FILE:mamba_synth>")
    add_custom_target(clap-validator
        COMMAND ${CLAP_VALIDATOR_BIN} validate "$<TARGET_FILE:mamba_synth>"
        DEPENDS mamba_synth
        COMMENT "Validating CLAP plugin with clap-validator"
        VERBATIM)
else()
    add_custom_target(clap-validator
        COMMAND ${CMAKE_COMMAND} -E echo "clap-validator not found; skipping plugin validation."
        COMMENT "clap-validator not installed"
        VERBATIM)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
