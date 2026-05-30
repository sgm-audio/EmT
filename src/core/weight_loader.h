#ifndef WEIGHT_LOADER_H
#define WEIGHT_LOADER_H

#include "config.h"
#include "core/ssm_block.h"
#include <cstddef>
#include <cstdint>

namespace mamba {

struct WeightFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t d_state;
    uint32_t d_model;
    uint32_t d_inner;
    uint32_t d_conv;
    uint32_t num_layers;
    uint32_t expansion_factor;
    uint32_t cond_dim;
    uint32_t reserved[4];
    uint64_t weight_data_size;
};

static constexpr uint32_t WEIGHT_MAGIC = 0x4D424D41;
static constexpr uint32_t WEIGHT_VERSION = 1;

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
};

class WeightLoader {
public:
    WeightLoader();
    ~WeightLoader();

    WeightLoader(const WeightLoader&) = delete;
    WeightLoader& operator=(const WeightLoader&) = delete;

    int load(const char* path);

    void unload();

    bool is_loaded() const { return mapped_data_ != nullptr; }

    const WeightFileHeader& header() const { return *header_; }

    bool validate_dimensions(uint32_t d_state, uint32_t d_model,
                             uint32_t d_inner, uint32_t d_conv,
                             uint32_t num_layers) const;

    int get_layer_pointers(uint32_t layer, WeightLayerPointers& out) const;

    uint32_t get_num_layers() const { return header_ ? header_->num_layers : 0; }

private:
    void* mapped_data_;
    size_t mapped_size_;
    int fd_;
    const WeightFileHeader* header_;
    const float* weight_data_;
    size_t weight_data_floats_;

    size_t floats_per_layer() const;
};

} // namespace mamba

#endif
