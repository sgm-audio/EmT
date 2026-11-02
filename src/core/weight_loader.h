#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "config.h"
#include "core/ssm_block.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mamba {

// ---------------------------------------------------------------------------
// On-disk weight file format
// ---------------------------------------------------------------------------
//
// See WEIGHT_FORMAT_SPEC.md for the authoritative spec.  The header is
// fixed at 64 bytes (cache-line aligned), little-endian.

struct WeightFileHeader {
    uint32_t magic;             // 0x4D424D41 ("MBMA")
    uint32_t version;           // 1
    uint32_t d_state;           // N
    uint32_t d_model;           // D
    uint32_t d_inner;           // E
    uint32_t d_conv;            // conv1d kernel
    uint32_t num_layers;        // L
    uint32_t expansion_factor;  // E / D
    uint32_t cond_dim;          // conditioning vector dim
    uint32_t quantization;      // 0 = FLOAT32, 1 = INT8 per-tensor, 2 = INT8 per-layer, 3 = FP16
    uint32_t q_metadata_size;   // size in bytes of q-metadata block (0 for FLOAT32)
    uint32_t flags;             // reserved (0)
    uint64_t weight_data_size;  // bytes of weight data after header (and q-metadata)
    uint64_t reserved;          // zero
};

static_assert(sizeof(WeightFileHeader) == 64,
              "WeightFileHeader must be exactly 64 bytes (cache-line aligned)");

static constexpr uint32_t WEIGHT_MAGIC   = 0x4D424D41;
static constexpr uint32_t WEIGHT_VERSION = 1;

enum class QuantFormat : uint32_t {
    Float32       = 0,
    Int8Tensor    = 1,
    Int8PerLayer  = 2,
    FP16          = 3,
};

// ---------------------------------------------------------------------------
// Layer weight views (zero-copy pointers into the mmap region)
// ---------------------------------------------------------------------------

// FLOAT32 views (one set of pointers per layer)
struct WeightLayerPointers {
    const float* A_log;
    const float* D_proj;
    const float* dt_bias;
    const float* dt_proj_w;
    const float* B_proj_w;
    const float* C_proj_w;
    const float* conv1d_w;
    const float* conv1d_b;
    const float* W_in;
    const float* W_out;
    const float* W_ssm;
    const float* b_in;
    const float* cond_proj_dt;
    const float* cond_proj_B;
    const float* cond_dt_bias;
    const float* cond_B_bias;
};

// INT8 views
struct WeightLayerPointersI8 {
    const int8_t* A_log;
    const int8_t* D_proj;
    const int8_t* dt_bias;
    const int8_t* dt_proj_w;
    const int8_t* B_proj_w;
    const int8_t* C_proj_w;
    const int8_t* conv1d_w;
    const int8_t* conv1d_b;
    const int8_t* W_in;
    const int8_t* W_out;
    const int8_t* W_ssm;
    const int8_t* b_in;
    const int8_t* cond_proj_dt;
    const int8_t* cond_proj_B;
    const int8_t* cond_dt_bias;
    const int8_t* cond_B_bias;
};

// FP16 views (stored as uint16_t; convert via fp16_to_fp32)
struct WeightLayerPointersF16 {
    const uint16_t* A_log;
    const uint16_t* D_proj;
    const uint16_t* dt_bias;
    const uint16_t* dt_proj_w;
    const uint16_t* B_proj_w;
    const uint16_t* C_proj_w;
    const uint16_t* conv1d_w;
    const uint16_t* conv1d_b;
    const uint16_t* W_in;
    const uint16_t* W_out;
    const uint16_t* W_ssm;
    const uint16_t* b_in;
    const uint16_t* cond_proj_dt;
    const uint16_t* cond_proj_B;
    const uint16_t* cond_dt_bias;
    const uint16_t* cond_B_bias;
};

// INT8 per-tensor dequant parameters (1 pair per file)
struct TensorQParam {
    float scale;
    int32_t zero_point;
};

// INT8 per-layer dequant parameters (1 pair per layer)
struct LayerQParam {
    float scale;
    int32_t zero_point;
};

// ---------------------------------------------------------------------------
// WeightLoader (mmap read-only, zero-copy, no allocations on hot path)
// ---------------------------------------------------------------------------

class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    // Resolve the path in this order:
    //   1. path argument (if non-null)
    //   2. MAMBA_WEIGHTS environment variable
    //   3. "models/test_model.bin" (relative to cwd)
    // Returns 0 on success or a negative error code.
    int load(const char* path = nullptr);

    void unload();

    bool is_loaded() const { return mapped_data_ != nullptr; }

    const WeightFileHeader& header() const { return *header_; }

    QuantFormat quant_format() const {
        return header_ ? static_cast<QuantFormat>(header_->quantization)
                       : QuantFormat::Float32;
    }

    // Verify the file's dimensions match the compiled-in config.h maxima.
    // Returns true if all dimensions fit (file dimensions <= MAX_*).
    bool fits_in_static_weights() const;

    // Static check (no loader instance required) for dimension validation.
    static bool fits_in_static_weights_with(const WeightFileHeader& hdr) {
        return hdr.d_state    > 0 && hdr.d_state    <= MAX_D_STATE  &&
               hdr.d_model    > 0 && hdr.d_model    <= MAX_D_MODEL  &&
               hdr.d_inner    > 0 && hdr.d_inner    <= MAX_D_INNER  &&
               hdr.d_conv     > 0 && hdr.d_conv     <= MAX_D_CONV   &&
               hdr.cond_dim   > 0 && hdr.cond_dim   <= COND_DIM     &&
               hdr.num_layers > 0 && hdr.num_layers <= MAX_LAYERS;
    }

    // True if (d_state, d_model, d_inner, d_conv, num_layers) match
    // exactly.  Use fits_in_static_weights() to allow the file to be a
    // strict subset of the static buffers.
    bool validate_dimensions(uint32_t d_state, uint32_t d_model,
                             uint32_t d_inner, uint32_t d_conv,
                             uint32_t num_layers) const;

    // FLOAT32 view. Returns -1 if format is not FLOAT32, -2 if layer OOB.
    int get_layer_pointers(uint32_t layer, WeightLayerPointers& out) const;

    // INT8 view. Returns -1 if format is not INT8, -2 if layer OOB.
    int get_layer_pointers_i8(uint32_t layer, WeightLayerPointersI8& out) const;

    // FP16 view. Returns -1 if format is not FP16, -2 if layer OOB.
    int get_layer_pointers_f16(uint32_t layer, WeightLayerPointersF16& out) const;

    // Per-tensor INT8 scale/zero_point. Valid only for QINT8_TENSOR.
    int get_tensor_qparam(TensorQParam& out) const;

    // Per-layer INT8 scale/zero_point. Valid only for QINT8_LAYER.
    int get_layer_qparam(uint32_t layer, LayerQParam& out) const;

    // Dequantize a single layer's INT8 weights to FLOAT32 in-place into the
    // caller-allocated SSMWeights struct.  The active dimensions are
    // bounded by config.h MAX_* values.
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
    const uint8_t* weight_data_;     // points past header + q-metadata
    size_t weight_data_bytes_;

    char source_path_[512];

    // Element size in bytes for the current format
    size_t elem_size() const;

    // Bytes per layer (depends on format)
    size_t bytes_per_layer() const;

    // Compute pointer to start of layer's weight block
    const uint8_t* layer_base(uint32_t layer) const;
};

// ---------------------------------------------------------------------------
// FP16 → FP32 IEEE 754 binary16 conversion (header-only, 5 lines of math)
// ---------------------------------------------------------------------------
//
// IEEE 754 binary16:
//   bit 15     : sign
//   bits 14-10 : biased exponent (bias 15)
//   bits 9-0   : mantissa (10 bits, implicit leading 1 unless subnormal/zero)
//
// Returns +/-0 for +/-0, +/-Inf for +/-Inf, NaN for NaN.  Subnormals are
// flushed to zero (FTZ) by design — matches SSMBlock::apply_ftz_daz().

inline float fp16_to_fp32(uint16_t h) {
    uint32_t sign     = (uint32_t)(h >> 15) & 0x1u;
    uint32_t exponent = (uint32_t)(h >> 10) & 0x1Fu;
    uint32_t mantissa = (uint32_t)h & 0x3FFu;
    uint32_t out;

    if (exponent == 0u) {
        // Zero or subnormal: FTZ → return signed zero
        out = sign << 31;
    } else if (exponent == 0x1Fu) {
        // Inf or NaN
        out = (sign << 31) | 0x7F800000u | (mantissa << 13);
    } else {
        // Normal: rebias exponent from 15 to 127
        out = (sign << 31) | (((exponent - 15u + 127u) & 0xFFu) << 23)
                            | (mantissa << 13);
    }
    float result;
    std::memcpy(&result, &out, sizeof(result));
    return result;
}

} // namespace mamba

#endif
