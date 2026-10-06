// Shared builders for weight-file tests: unique temp paths, MAX-layout float
// math, and minimal valid .bin writers.
//
// Header-only with no GTest dependency, so benchmarks and tools can reuse it.
// Include as "fixtures/weight_fixture.h" (tests/ is on the include path of
// every test target via the test_fixtures interface library).
//
// This mirrors scripts/emt_format.py on the C++ side: the file-layout math
// here (in_proj_out_for / floats_per_layer_for) must agree with the Python
// module and with WeightLoader::bytes_per_layer().
#ifndef EMT_TEST_WEIGHT_FIXTURE_H
#define EMT_TEST_WEIGHT_FIXTURE_H

#include "core/weight_loader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
#else
#include <unistd.h>
#endif

namespace mamba::test {

inline int test_pid() {
#if defined(_WIN32)
    return static_cast<int>(GetCurrentProcessId());
#else
    return static_cast<int>(getpid());
#endif
}

inline void remove_test_file(const char* path) {
#if defined(_WIN32)
    _unlink(path);
#else
    unlink(path);
#endif
}

// RAII temp path for weight files. Unlinks on destruction; movable so
// factory helpers can return it by value.
class TempWeightFile {
public:
    explicit TempWeightFile(const char* prefix) : owned_(true) {
        std::filesystem::path p = std::filesystem::temp_directory_path() /
            (std::string(prefix) + std::to_string(test_pid()) + ".bin");
        std::strncpy(path, p.string().c_str(), sizeof(path) - 1);
        path[sizeof(path) - 1] = '\0';
    }

    ~TempWeightFile() {
        if (owned_) remove_test_file(path);
    }

    TempWeightFile(const TempWeightFile&) = delete;
    TempWeightFile& operator=(const TempWeightFile&) = delete;

    TempWeightFile(TempWeightFile&& other) noexcept : owned_(other.owned_) {
        std::memcpy(path, other.path, sizeof(path));
        other.owned_ = false;
    }
    TempWeightFile& operator=(TempWeightFile&& other) noexcept {
        if (this != &other) {
            if (owned_) remove_test_file(path);
            std::memcpy(path, other.path, sizeof(path));
            owned_ = other.owned_;
            other.owned_ = false;
        }
        return *this;
    }

    char path[512];

private:
    bool owned_;
};

// Fused-projection row count in the *file* layout (MAX_*-padded).
// Row groups: z + x + B + C + (dd_dt, dd_A, trap) + angles.
inline size_t in_proj_out_for(uint32_t d_inner, uint32_t nheads,
                              uint32_t d_state) {
    return static_cast<size_t>(d_inner) * 2 +
           static_cast<size_t>(nheads) * MAX_MIMO_RANK * d_state * 2 +
           static_cast<size_t>(nheads) * 3 + MAX_NUM_ROPE_ANGLES;
}

// Float32 element count per layer in the MAX_*-padded file layout.
inline size_t floats_per_layer_for(uint32_t ds, uint32_t dm, uint32_t di,
                                   uint32_t nh, uint32_t hd, uint32_t cd) {
    const size_t in_proj_out = in_proj_out_for(di, nh, ds);
    return in_proj_out * dm +                // in_proj_w
           nh +                              // dt_bias
           static_cast<size_t>(nh) * MAX_MIMO_RANK * ds +  // B_bias
           static_cast<size_t>(nh) * MAX_MIMO_RANK * ds +  // C_bias
           ds +                              // B_norm_weight
           ds +                              // C_norm_weight
           static_cast<size_t>(nh) * MAX_MIMO_RANK * hd +  // mimo_x
           static_cast<size_t>(nh) * MAX_MIMO_RANK * hd +  // mimo_z
           static_cast<size_t>(nh) * MAX_MIMO_RANK * hd +  // mimo_o
           nh +                              // D_proj
           static_cast<size_t>(dm) * di +    // out_proj_w
           static_cast<size_t>(2) * dm * cd +  // film_proj_w
           static_cast<size_t>(2) * nh * cd;   // dt_film_proj_w
}

struct WeightFileSpec {
    uint32_t d_state = MAX_D_STATE;
    uint32_t d_model = MAX_D_MODEL;
    uint32_t d_inner = MAX_D_INNER;
    uint32_t nheads = MAX_NHEADS;
    uint32_t headdim = MAX_HEADDIM;
    uint32_t cond_dim = COND_DIM;
    uint32_t num_layers = 1;
    // mimo_rank defaults to 0 = legacy file written before the field existed
    // (WeightLoader treats 0 as rank 1). Pass > 1 to exercise the D3 refusal.
    uint32_t mimo_rank = 0;
    float fill = 0.1f;       // payload fill (small non-zero)
    float first = -1.0f;     // recognizable first element…
    bool mark_first = true;  // …written when true
};

// Write a minimal valid float32 .bin: header + `num_layers` of `fill`
// payload. Returns false on I/O error.
inline bool write_weight_file(const char* path, const WeightFileSpec& spec) {
    const size_t fpl = floats_per_layer_for(spec.d_state, spec.d_model,
                                            spec.d_inner, spec.nheads,
                                            spec.headdim, spec.cond_dim);
    const size_t total_floats = fpl * spec.num_layers;

    FILE* f = std::fopen(path, "wb");
    if (!f) return false;

    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = spec.d_state;
    hdr.d_model = spec.d_model;
    hdr.d_inner = spec.d_inner;
    hdr.nheads = spec.nheads;
    hdr.num_layers = spec.num_layers;
    hdr.headdim = spec.headdim;
    hdr.cond_dim = spec.cond_dim;
    hdr.mimo_rank = spec.mimo_rank;
    hdr.weight_data_size = static_cast<uint64_t>(total_floats * sizeof(float));
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    std::vector<float> data(total_floats, spec.fill);
    if (spec.mark_first && !data.empty()) data[0] = spec.first;
    std::fwrite(data.data(), sizeof(float), total_floats, f);
    std::fclose(f);
    return true;
}

}  // namespace mamba::test

#endif  // EMT_TEST_WEIGHT_FIXTURE_H
