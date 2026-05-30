cmake_minimum_required(VERSION 3.20)
project(MambaSSM VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")

string(APPEND CMAKE_CXX_FLAGS_RELEASE " -fno-exceptions -fno-rtti -fvisibility=hidden -Wall -Wextra -Wpedantic")

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()

include(FetchContent)

FetchContent_Declare(
googletest
GIT_REPOSITORY https://github.com/google/googletest.git
GIT_TAG v1.14.0
)
set(gtest_force_shared_crt ON)
set(BUILD_GMOCK OFF)
set(INSTALL_GTEST OFF)
FetchContent_MakeAvailable(googletest)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/clap/include)

add_subdirectory(src/core)
add_subdirectory(src/common)
add_subdirectory(src/ara)
add_subdirectory(src/clap)
add_subdirectory(tests)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
