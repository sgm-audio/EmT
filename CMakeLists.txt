cmake_minimum_required(VERSION 3.20)
project(EmT VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /arch:AVX2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /Zi")
    set(_MAMBA_WARN_FLAGS /W4 /permissive- /wd4324 /wd4996 /D_CRT_SECURE_NO_WARNINGS)
    set(_MAMBA_NO_EXCEPTIONS_RTTI_FLAGS /GR-)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mavx2 -mfma -fno-math-errno -fno-trapping-math -fno-reciprocal-math -fno-signaling-nans -fcx-limited-range")
    set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address")
    set(_MAMBA_WARN_FLAGS -Wall -Wextra -Wpedantic)
    set(_MAMBA_NO_EXCEPTIONS_RTTI_FLAGS -fno-exceptions -fno-rtti -fvisibility=hidden)
endif()

# Apply extra flags via add_compile_options so they remain separate tokens.
add_compile_options(${_MAMBA_NO_EXCEPTIONS_RTTI_FLAGS} ${_MAMBA_WARN_FLAGS})

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
set(CMAKE_OSX_DEPLOYMENT_TARGET "13.0")
set(CMAKE_OSX_ARCHITECTURES "arm64;x86_64")
endif()

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

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/extern/clap/include)

add_subdirectory(src/core)
add_subdirectory(src/common)
add_subdirectory(src/ara)
add_subdirectory(src/clap)
enable_testing()
add_subdirectory(tests)

# Custom target: clap-validator
# Runs the CLAP plugin validator against the built .clap bundle.
# If clap-validator is not found, prints installation instructions.
add_custom_target(clap-validator
    COMMAND ${CMAKE_COMMAND} -E echo "Running clap-validator..."
    COMMAND bash -c "\
        if command -v clap-validator &> /dev/null; then \
            clap-validator ${CMAKE_BINARY_DIR}/src/clap/MambaSynth.clap; \
        else \
            echo 'clap-validator not found in PATH'; \
            echo 'Install from: https://github.com/free-audio/clap'; \
            echo '  git clone https://github.com/free-audio/clap.git'; \
            echo '  cd clap && mkdir build && cd build'; \
            echo '  cmake .. -DCMAKE_BUILD_TYPE=Release'; \
            echo '  cmake --build . --target clap-validator'; \
            echo '  sudo cp clap-validator /usr/local/bin/'; \
            exit 0; \
        fi"
    DEPENDS mamba_synth
    COMMENT "Validating CLAP plugin with clap-validator"
    VERBATIM)

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
