#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "config.h"
#include "core/ssm_block.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mamba {

// ---------------------------------------------------------------------------
// On-disk weight file format (Mamba-3)
// ---------------------------------------------------------------------------
//
// Header is 64 bytes (cache-line aligned), little-endian.
// Version 2 for Mamba-3: no conv1d, MIMO+multi-head+RMSNorm+FiLM layout.

struct WeightFileHeader {
    uint32_t magic;             // 0x4D424D41 ("MBMA")
    uint32_t version;           // 2 (Mamba-3 format)
    uint32_t d_state;           // N (state dimension)
    uint32_t d_model;           // D (embedding dimension)
    uint32_t d_inner;           // E (expanded dimension = 2*D)
    uint32_t nheads;            // H (number of heads, previously d_conv slot)
    uint32_t num_layers;        // L
    uint32_t headdim;           // P (head dimension = E/H)
    uint32_t cond_dim;          // conditioning vector dim
    uint32_t quantization;      // 0 = FLOAT32, 1 = INT8 per-tensor, 2 = INT8 per-layer, 3 = FP16
    uint32_t q_metadata_size;   // size in bytes of q-metadata block (0 for FLOAT32)
    uint32_t flags;             // reserved (0)
    uint64_t weight_data_size;  // bytes of weight data after header (and q-metadata)
    uint32_t mimo_rank;         // version 2 only: 1 = rank-1 SISO (the only rank
                                // this runtime implements).  0 = legacy file
                                // written before this field existed; treated as
                                // rank 1.  Values > 1 are refused by load()
                                // (deviation D3).
    uint32_t reserved;          // zero
};

static_assert(sizeof(WeightFileHeader) == 64,
              "WeightFileHeader must be exactly 64 bytes (cache-line aligned)");

static constexpr uint32_t WEIGHT_MAGIC   = 0x4D424D41;
static constexpr uint32_t WEIGHT_VERSION = 2;  // Mamba-3 format

enum class QuantFormat : uint32_t {
    Float32       = 0,
    Int8Tensor    = 1,
    Int8PerLayer  = 2,
    FP16          = 3,
};

// ---------------------------------------------------------------------------
// Layer weight views (zero-copy pointers into the mmap region)
// ---------------------------------------------------------------------------
// Mamba-3: 13 arrays per layer (down from 16 in Mamba-1).
// No conv1d. All projections fused into in_proj.

struct WeightLayerPointers {
    const float* in_proj_w;       // [in_proj_out, d_model]
    const float* dt_bias;         // [nheads]
    const float* B_bias;          // [nheads, mimo_rank, d_state]
    const float* C_bias;          // [nheads, mimo_rank, d_state]
    const float* B_norm_weight;   // [d_state]
    const float* C_norm_weight;   // [d_state]
    const float* mimo_x;          // [nheads, mimo_rank, headdim]
    const float* mimo_z;          // [nheads, mimo_rank, headdim]
    const float* mimo_o;          // [nheads, mimo_rank, headdim]
    const float* D_proj;          // [nheads]
    const float* out_proj_w;      // [d_model, d_inner]
    const float* film_proj_w;     // [2 * d_model, d_cond]
    const float* dt_film_proj_w;  // [2 * nheads, d_cond]
};

struct WeightLayerPointersI8 {
    const int8_t* in_proj_w;
    const int8_t* dt_bias;
    const int8_t* B_bias;
    const int8_t* C_bias;
    const int8_t* B_norm_weight;
    const int8_t* C_norm_weight;
    const int8_t* mimo_x;
    const int8_t* mimo_z;
    const int8_t* mimo_o;
    const int8_t* D_proj;
    const int8_t* out_proj_w;
    const int8_t* film_proj_w;
    const int8_t* dt_film_proj_w;
};

struct WeightLayerPointersF16 {
    const uint16_t* in_proj_w;
    const uint16_t* dt_bias;
    const uint16_t* B_bias;
    const uint16_t* C_bias;
    const uint16_t* B_norm_weight;
    const uint16_t* C_norm_weight;
    const uint16_t* mimo_x;
    const uint16_t* mimo_z;
    const uint16_t* mimo_o;
    const uint16_t* D_proj;
    const uint16_t* out_proj_w;
    const uint16_t* film_proj_w;
    const uint16_t* dt_film_proj_w;
};

struct TensorQParam {
    float scale;
    int32_t zero_point;
};

struct LayerQParam {
    float scale;
    int32_t zero_point;
};

// ---------------------------------------------------------------------------
// WeightLoader
// ---------------------------------------------------------------------------

class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    int load(const char* path = nullptr);
    void unload();

    bool is_loaded() const { return mapped_data_ != nullptr; }
    const WeightFileHeader& header() const { return *header_; }

    QuantFormat quant_format() const {
        return header_ ? static_cast<QuantFormat>(header_->quantization)
                       : QuantFormat::Float32;
    }

    bool fits_in_static_weights() const;

    // Return codes from load():
    //    0  ok
    //  -10  file smaller than the header
    //  -11  bad magic
    //  -12  unsupported version
    //  -13  truncated / undersized weight data
    //  -14  dimension exceeds the compiled MAX_* limits
    //  -15  unsupported quantization format
    //  -16  mimo_rank > 1 (deviation D3: reserved in the format, not implemented)
    static bool fits_in_static_weights_with(const WeightFileHeader& hdr) {
        return hdr.d_state    > 0 && hdr.d_state    <= MAX_D_STATE  &&
               hdr.d_model    > 0 && hdr.d_model    <= MAX_D_MODEL  &&
               hdr.d_inner    > 0 && hdr.d_inner    <= MAX_D_INNER  &&
               hdr.nheads     > 0 && hdr.nheads     <= MAX_NHEADS   &&
               hdr.headdim    > 0 && hdr.headdim    <= MAX_HEADDIM  &&
               hdr.cond_dim   > 0 && hdr.cond_dim   <= COND_DIM     &&
               hdr.num_layers > 0 && hdr.num_layers <= MAX_LAYERS   &&
               // Deviation D3: a rank-R file must not look loadable.
               (hdr.version < 2 || hdr.mimo_rank <= 1);
    }

    bool validate_dimensions(uint32_t d_state, uint32_t d_model,
                             uint32_t d_inner, uint32_t nheads,
                             uint32_t headdim, uint32_t num_layers) const;

    int get_layer_pointers(uint32_t layer, WeightLayerPointers& out) const;
    int get_layer_pointers_i8(uint32_t layer, WeightLayerPointersI8& out) const;
    int get_layer_pointers_f16(uint32_t layer, WeightLayerPointersF16& out) const;

    int get_tensor_qparam(TensorQParam& out) const;
    int get_layer_qparam(uint32_t layer, LayerQParam& out) const;

    int dequantize_layer_to(uint32_t layer, SSMWeights& out_f32) const;

    uint32_t get_num_layers() const {
        return header_ ? header_->num_layers : 0;
    }

    const char* source_path() const { return source_path_; }

private:
    void* mapped_data_;
    size_t mapped_size_;
    int fd_;
    const WeightFileHeader* header_;
    const uint8_t* weight_data_;
    size_t weight_data_bytes_;

    char source_path_[512];

    size_t elem_size() const;
    size_t bytes_per_layer() const;
    const uint8_t* layer_base(uint32_t layer) const;
};

// ---------------------------------------------------------------------------
// FP16 → FP32
// ---------------------------------------------------------------------------

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exponent = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)h & 0x3FFu;
    uint32_t out;

    if (exponent == 0u) {
        out = sign << 31;
    } else if (exponent == 0x1Fu) {
        out = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        out = (sign << 31) | (((exponent - 15u + 127u) & 0xFFu) << 23)
                            | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

} // namespace mamba

#endif
