#include "core/weight_loader.h"
#include <cstring>
#include <cstdlib>
#include <cstdio>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mamba {

namespace {

struct LayerQMetaRaw {
    float    scale;
    int32_t  zero_point;
    uint8_t  pad[8];
};
static_assert(sizeof(LayerQMetaRaw) == 16, "LayerQMetaRaw must be 16 bytes");

struct TensorQMetaRaw {
    float    scale;
    int32_t  zero_point;
    uint8_t  pad[8];
};
static_assert(sizeof(TensorQMetaRaw) == 16, "TensorQMetaRaw must be 16 bytes");

bool resolve_path(const char* path, char* out, size_t out_size) {
    if (path && *path) {
        std::snprintf(out, out_size, "%s", path);
        return true;
    }
    const char* env = std::getenv("EMT_WEIGHTS");
    if (!env || !*env) env = std::getenv("MAMBA_WEIGHTS");
    if (env && *env) {
        std::snprintf(out, out_size, "%s", env);
        return true;
    }
    std::snprintf(out, out_size, "models/test_model.bin");
    return false;
}

} // namespace

WeightLoader::WeightLoader()
    : mapped_data_(nullptr), mapped_size_(0), fd_(-1),
      header_(nullptr), weight_data_(nullptr), weight_data_bytes_(0) {
    source_path_[0] = '\0';
}

WeightLoader::~WeightLoader() {
    unload();
}

int WeightLoader::load(const char* path) {
    unload();

    char resolved[512];
    resolve_path(path, resolved, sizeof(resolved));
    std::snprintf(source_path_, sizeof(source_path_), "%s", resolved);

#ifdef _WIN32
    HANDLE hFile = CreateFileA(resolved, GENERIC_READ, FILE_SHARE_READ,
                               nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_READONLY, nullptr);
    if (hFile == INVALID_HANDLE_VALUE) return -1;

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(hFile, &file_size)) { CloseHandle(hFile); return -2; }
    mapped_size_ = static_cast<size_t>(file_size.QuadPart);

    HANDLE hMap = CreateFileMappingA(hFile, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (!hMap) { CloseHandle(hFile); return -3; }

    mapped_data_ = MapViewOfFile(hMap, FILE_MAP_READ, 0, 0, 0);
    CloseHandle(hMap);
    CloseHandle(hFile);

    if (!mapped_data_) return -4;
    fd_ = -1;
#else
    fd_ = open(resolved, O_RDONLY);
    if (fd_ < 0) return -1;

    struct stat st;
    if (fstat(fd_, &st) < 0) { close(fd_); fd_ = -1; return -2; }
    mapped_size_ = static_cast<size_t>(st.st_size);

    mapped_data_ = mmap(nullptr, mapped_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_data_ == MAP_FAILED) {
        mapped_data_ = nullptr;
        close(fd_); fd_ = -1;
        return -3;
    }
#endif

    if (mapped_size_ < sizeof(WeightFileHeader)) {
        unload();
        return -10;
    }

    header_ = static_cast<const WeightFileHeader*>(mapped_data_);

    // Accept both version 1 (Mamba-1) and version 2 (Mamba-3)
    if (header_->magic != WEIGHT_MAGIC) {
        unload();
        return -11;
    }

    if (header_->version != 1 && header_->version != 2) {
        unload();
        return -12;
    }

    // Mamba-3 dimension validation
    if (header_->d_state   == 0 || header_->d_state   > MAX_D_STATE)  { unload(); return -14; }
    if (header_->d_model   == 0 || header_->d_model   > MAX_D_MODEL)  { unload(); return -14; }
    if (header_->d_inner   == 0 || header_->d_inner   > MAX_D_INNER)  { unload(); return -14; }
    if (header_->nheads    == 0 || header_->nheads    > MAX_NHEADS)   { unload(); return -14; }
    if (header_->headdim   == 0 || header_->headdim   > MAX_HEADDIM)  { unload(); return -14; }
    if (header_->cond_dim  == 0 || header_->cond_dim  > COND_DIM)     { unload(); return -14; }
    if (header_->num_layers== 0 || header_->num_layers > MAX_LAYERS) { unload(); return -14; }

    // Validate quantization format
    if (header_->quantization > static_cast<uint32_t>(QuantFormat::FP16)) {
        unload();
        return -15;
    }

    // Compute offset of weight data (header + q-metadata)
    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    size_t q_meta_size = header_->q_metadata_size;
    weight_data_ = base + sizeof(WeightFileHeader) + q_meta_size;
    weight_data_bytes_ = static_cast<size_t>(header_->weight_data_size);

    // Bounds check
    size_t needed = sizeof(WeightFileHeader) + q_meta_size + weight_data_bytes_;
    if (needed > mapped_size_) {
        unload();
        return -13;
    }

    // Per-layer size validation
    size_t expected_bpl = bytes_per_layer();
    size_t expected_total = expected_bpl * header_->num_layers;
    if (weight_data_bytes_ < expected_total) {
        unload();
        return -13;
    }

    return 0;
}

void WeightLoader::unload() {
    if (mapped_data_) {
#ifdef _WIN32
        UnmapViewOfFile(mapped_data_);
#else
        munmap(mapped_data_, mapped_size_);
#endif
        mapped_data_ = nullptr;
    }

    if (fd_ >= 0) {
#ifndef _WIN32
        close(fd_);
#endif
        fd_ = -1;
    }

    mapped_size_ = 0;
    header_ = nullptr;
    weight_data_ = nullptr;
    weight_data_bytes_ = 0;
    source_path_[0] = '\0';
}

bool WeightLoader::fits_in_static_weights() const {
    return validate_dimensions(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER,
                               MAX_NHEADS, MAX_HEADDIM, MAX_LAYERS);
}

bool WeightLoader::validate_dimensions(uint32_t d_state, uint32_t d_model,
                                        uint32_t d_inner, uint32_t nheads,
                                        uint32_t headdim, uint32_t num_layers) const {
    if (!header_) return false;
    return header_->d_state    == d_state  &&
           header_->d_model    == d_model  &&
           header_->d_inner    == d_inner  &&
           header_->nheads     == nheads   &&
           header_->headdim    == headdim  &&
           header_->num_layers == num_layers;
}

size_t WeightLoader::elem_size() const {
    switch (quant_format()) {
        case QuantFormat::Float32:      return 4;
        case QuantFormat::Int8Tensor:
        case QuantFormat::Int8PerLayer: return 1;
        case QuantFormat::FP16:         return 2;
    }
    return 4;
}

size_t WeightLoader::bytes_per_layer() const {
    if (!header_) return 0;
    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t nh = header_->nheads;       // nheads
    uint32_t hd = header_->headdim;      // headdim
    uint32_t cd = header_->cond_dim;

    // Mamba-3 layout: in_proj stores all projections
    size_t in_proj_out = di * 2                    // z + x
                       + nh * MAX_MIMO_RANK * ds * 2  // B + C
                       + nh * 3                     // dd_dt + dd_A + trap
                       + MAX_NUM_ROPE_ANGLES;       // angles

    size_t elems = 0;
    elems += in_proj_out * dm;              // in_proj_w
    elems += nh;                             // dt_bias
    elems += nh * MAX_MIMO_RANK * ds;        // B_bias
    elems += nh * MAX_MIMO_RANK * ds;        // C_bias
    elems += ds;                             // B_norm_weight
    elems += ds;                             // C_norm_weight
    elems += nh * MAX_MIMO_RANK * hd;        // mimo_x
    elems += nh * MAX_MIMO_RANK * hd;        // mimo_z
    elems += nh * MAX_MIMO_RANK * hd;        // mimo_o
    elems += nh;                             // D_proj
    elems += dm * di;                        // out_proj_w
    elems += 2 * dm * cd;                    // film_proj_w
    elems += 2 * nh * cd;                    // dt_film_proj_w
    return elems * elem_size();
}

const uint8_t* WeightLoader::layer_base(uint32_t layer) const {
    if (!weight_data_ || !header_) return nullptr;
    if (layer >= header_->num_layers) return nullptr;
    return weight_data_ + layer * bytes_per_layer();
}

int WeightLoader::get_layer_pointers(uint32_t layer,
                                     WeightLayerPointers& out) const {
    if (!header_ || !weight_data_) return -1;
    if (quant_format() != QuantFormat::Float32) return -1;
    if (layer >= header_->num_layers) return -2;

    const uint8_t* base = layer_base(layer);
    if (!base) return -1;
    const float* fbase = reinterpret_cast<const float*>(base);

    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t nh = header_->nheads;
    uint32_t hd = header_->headdim;
    uint32_t cd = header_->cond_dim;

    size_t in_proj_out = di * 2
                       + nh * MAX_MIMO_RANK * ds * 2
                       + nh * 3
                       + MAX_NUM_ROPE_ANGLES;

    size_t off = 0;
    out.in_proj_w     = fbase + off; off += in_proj_out * dm;
    out.dt_bias       = fbase + off; off += nh;
    out.B_bias        = fbase + off; off += nh * MAX_MIMO_RANK * ds;
    out.C_bias        = fbase + off; off += nh * MAX_MIMO_RANK * ds;
    out.B_norm_weight = fbase + off; off += ds;
    out.C_norm_weight = fbase + off; off += ds;
    out.mimo_x        = fbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_z        = fbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_o        = fbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.D_proj        = fbase + off; off += nh;
    out.out_proj_w    = fbase + off; off += dm * di;
    out.film_proj_w   = fbase + off; off += 2 * dm * cd;
    out.dt_film_proj_w = fbase + off;
    return 0;
}

int WeightLoader::get_layer_pointers_i8(uint32_t layer,
                                         WeightLayerPointersI8& out) const {
    if (!header_ || !weight_data_) return -1;
    if (quant_format() != QuantFormat::Int8Tensor &&
        quant_format() != QuantFormat::Int8PerLayer) return -1;
    if (layer >= header_->num_layers) return -2;

    const uint8_t* base = layer_base(layer);
    if (!base) return -1;
    const int8_t* ibase = reinterpret_cast<const int8_t*>(base);

    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t nh = header_->nheads;
    uint32_t hd = header_->headdim;
    uint32_t cd = header_->cond_dim;

    size_t in_proj_out = di * 2
                       + nh * MAX_MIMO_RANK * ds * 2
                       + nh * 3
                       + MAX_NUM_ROPE_ANGLES;

    size_t off = 0;
    out.in_proj_w     = ibase + off; off += in_proj_out * dm;
    out.dt_bias       = ibase + off; off += nh;
    out.B_bias        = ibase + off; off += nh * MAX_MIMO_RANK * ds;
    out.C_bias        = ibase + off; off += nh * MAX_MIMO_RANK * ds;
    out.B_norm_weight = ibase + off; off += ds;
    out.C_norm_weight = ibase + off; off += ds;
    out.mimo_x        = ibase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_z        = ibase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_o        = ibase + off; off += nh * MAX_MIMO_RANK * hd;
    out.D_proj        = ibase + off; off += nh;
    out.out_proj_w    = ibase + off; off += dm * di;
    out.film_proj_w   = ibase + off; off += 2 * dm * cd;
    out.dt_film_proj_w = ibase + off;
    return 0;
}

int WeightLoader::get_layer_pointers_f16(uint32_t layer,
                                          WeightLayerPointersF16& out) const {
    if (!header_ || !weight_data_) return -1;
    if (quant_format() != QuantFormat::FP16) return -1;
    if (layer >= header_->num_layers) return -2;

    const uint8_t* base = layer_base(layer);
    if (!base) return -1;
    const uint16_t* hbase = reinterpret_cast<const uint16_t*>(base);

    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t nh = header_->nheads;
    uint32_t hd = header_->headdim;
    uint32_t cd = header_->cond_dim;

    size_t in_proj_out = di * 2
                       + nh * MAX_MIMO_RANK * ds * 2
                       + nh * 3
                       + MAX_NUM_ROPE_ANGLES;

    size_t off = 0;
    out.in_proj_w     = hbase + off; off += in_proj_out * dm;
    out.dt_bias       = hbase + off; off += nh;
    out.B_bias        = hbase + off; off += nh * MAX_MIMO_RANK * ds;
    out.C_bias        = hbase + off; off += nh * MAX_MIMO_RANK * ds;
    out.B_norm_weight = hbase + off; off += ds;
    out.C_norm_weight = hbase + off; off += ds;
    out.mimo_x        = hbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_z        = hbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.mimo_o        = hbase + off; off += nh * MAX_MIMO_RANK * hd;
    out.D_proj        = hbase + off; off += nh;
    out.out_proj_w    = hbase + off; off += dm * di;
    out.film_proj_w   = hbase + off; off += 2 * dm * cd;
    out.dt_film_proj_w = hbase + off;
    return 0;
}

int WeightLoader::get_tensor_qparam(TensorQParam& out) const {
    if (!header_) return -1;
    if (quant_format() != QuantFormat::Int8Tensor) return -2;

    if (header_->q_metadata_size < sizeof(TensorQMetaRaw)) return -3;
    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    const TensorQMetaRaw* m = reinterpret_cast<const TensorQMetaRaw*>(
        base + sizeof(WeightFileHeader));
    out.scale = m->scale;
    out.zero_point = m->zero_point;
    return 0;
}

int WeightLoader::get_layer_qparam(uint32_t layer, LayerQParam& out) const {
    if (!header_) return -1;
    if (quant_format() != QuantFormat::Int8PerLayer) return -2;
    if (layer >= header_->num_layers) return -3;

    size_t needed = sizeof(LayerQMetaRaw) * header_->num_layers;
    if (header_->q_metadata_size < needed) return -4;

    const uint8_t* base = static_cast<const uint8_t*>(mapped_data_);
    const LayerQMetaRaw* m = reinterpret_cast<const LayerQMetaRaw*>(
        base + sizeof(WeightFileHeader));
    out.scale = m[layer].scale;
    out.zero_point = m[layer].zero_point;
    return 0;
}

int WeightLoader::dequantize_layer_to(uint32_t layer, SSMWeights& out_f32) const {
    if (!header_ || !weight_data_) return -1;
    if (layer >= header_->num_layers) return -2;

    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t nh = header_->nheads;
    uint32_t hd = header_->headdim;
    uint32_t cd = header_->cond_dim;

    size_t in_proj_out = di * 2
                       + nh * MAX_MIMO_RANK * ds * 2
                       + nh * 3
                       + MAX_NUM_ROPE_ANGLES;

    std::memset(&out_f32, 0, sizeof(out_f32));

    // Helper lambdas for row-by-row copy to avoid MAX_* stride corruption
    auto copy_in_proj = [&](auto get_val) {
        for (size_t r = 0; r < in_proj_out; ++r)
            for (size_t c = 0; c < dm; ++c)
                out_f32.in_proj_w[r][c] = get_val(r, c);
    };
    auto copy_1d = [&](auto& dst, auto get_val, size_t n) {
        for (size_t i = 0; i < n; ++i)
            dst[i] = get_val(i);
    };
    auto copy_mimo = [&](auto& dst, auto get_val) {
        for (size_t h = 0; h < nh; ++h)
            for (size_t r = 0; r < MAX_MIMO_RANK; ++r)
                for (size_t s = 0; s < ds; ++s)
                    dst[h][r][s] = get_val((h * MAX_MIMO_RANK + r) * ds + s);
    };
    auto copy_mimo_hd = [&](auto& dst, auto get_val) {
        for (size_t h = 0; h < nh; ++h)
            for (size_t r = 0; r < MAX_MIMO_RANK; ++r)
                for (size_t s = 0; s < hd; ++s)
                    dst[h][r][s] = get_val((h * MAX_MIMO_RANK + r) * hd + s);
    };
    auto copy_out_proj = [&](auto get_val) {
        for (size_t r = 0; r < dm; ++r)
            for (size_t c = 0; c < di; ++c)
                out_f32.out_proj_w[r][c] = get_val(r * di + c);
    };
    auto copy_film = [&](auto get_val) {
        for (size_t r = 0; r < 2 * dm; ++r)
            for (size_t c = 0; c < cd; ++c)
                out_f32.film_proj_w[r][c] = get_val(r * cd + c);
    };

    switch (quant_format()) {
        case QuantFormat::Float32: {
            WeightLayerPointers p;
            if (get_layer_pointers(layer, p) != 0) return -3;

            copy_in_proj([&](size_t r, size_t c) { return p.in_proj_w[r * dm + c]; });
            copy_1d(out_f32.dt_bias, [&](size_t i) { return p.dt_bias[i]; }, nh);
            copy_mimo(out_f32.B_bias, [&](size_t i) { return p.B_bias[i]; });
            copy_mimo(out_f32.C_bias, [&](size_t i) { return p.C_bias[i]; });
            copy_1d(out_f32.B_norm_weight, [&](size_t i) { return p.B_norm_weight[i]; }, ds);
            copy_1d(out_f32.C_norm_weight, [&](size_t i) { return p.C_norm_weight[i]; }, ds);
            copy_mimo_hd(out_f32.mimo_x, [&](size_t i) { return p.mimo_x[i]; });
            copy_mimo_hd(out_f32.mimo_z, [&](size_t i) { return p.mimo_z[i]; });
            copy_mimo_hd(out_f32.mimo_o, [&](size_t i) { return p.mimo_o[i]; });
            copy_1d(out_f32.D_proj, [&](size_t i) { return p.D_proj[i]; }, nh);
            copy_out_proj([&](size_t i) { return p.out_proj_w[i]; });
            copy_film([&](size_t i) { return p.film_proj_w[i]; });
            for (size_t r = 0; r < 2 * nh; ++r)
                for (size_t c = 0; c < cd; ++c)
                    out_f32.dt_film_proj_w[r][c] = p.dt_film_proj_w[r * cd + c];
            return 0;
        }
        case QuantFormat::Int8Tensor: {
            WeightLayerPointersI8 p;
            if (get_layer_pointers_i8(layer, p) != 0) return -3;
            TensorQParam qp;
            if (get_tensor_qparam(qp) != 0) return -4;

            auto deq = [&](int8_t v) { return qp.scale * (static_cast<float>(v) - static_cast<float>(qp.zero_point)); };

            copy_in_proj([&](size_t r, size_t c) { return deq(p.in_proj_w[r * dm + c]); });
            copy_1d(out_f32.dt_bias, [&](size_t i) { return deq(p.dt_bias[i]); }, nh);
            copy_mimo(out_f32.B_bias, [&](size_t i) { return deq(p.B_bias[i]); });
            copy_mimo(out_f32.C_bias, [&](size_t i) { return deq(p.C_bias[i]); });
            copy_1d(out_f32.B_norm_weight, [&](size_t i) { return deq(p.B_norm_weight[i]); }, ds);
            copy_1d(out_f32.C_norm_weight, [&](size_t i) { return deq(p.C_norm_weight[i]); }, ds);
            copy_mimo_hd(out_f32.mimo_x, [&](size_t i) { return deq(p.mimo_x[i]); });
            copy_mimo_hd(out_f32.mimo_z, [&](size_t i) { return deq(p.mimo_z[i]); });
            copy_mimo_hd(out_f32.mimo_o, [&](size_t i) { return deq(p.mimo_o[i]); });
            copy_1d(out_f32.D_proj, [&](size_t i) { return deq(p.D_proj[i]); }, nh);
            copy_out_proj([&](size_t i) { return deq(p.out_proj_w[i]); });
            copy_film([&](size_t i) { return deq(p.film_proj_w[i]); });
            for (size_t r = 0; r < 2 * nh; ++r)
                for (size_t c = 0; c < cd; ++c)
                    out_f32.dt_film_proj_w[r][c] = deq(p.dt_film_proj_w[r * cd + c]);
            return 0;
        }
        case QuantFormat::Int8PerLayer: {
            WeightLayerPointersI8 p;
            if (get_layer_pointers_i8(layer, p) != 0) return -3;
            LayerQParam qp;
            if (get_layer_qparam(layer, qp) != 0) return -4;

            auto deq = [&](int8_t v) { return qp.scale * (static_cast<float>(v) - static_cast<float>(qp.zero_point)); };

            copy_in_proj([&](size_t r, size_t c) { return deq(p.in_proj_w[r * dm + c]); });
            copy_1d(out_f32.dt_bias, [&](size_t i) { return deq(p.dt_bias[i]); }, nh);
            copy_mimo(out_f32.B_bias, [&](size_t i) { return deq(p.B_bias[i]); });
            copy_mimo(out_f32.C_bias, [&](size_t i) { return deq(p.C_bias[i]); });
            copy_1d(out_f32.B_norm_weight, [&](size_t i) { return deq(p.B_norm_weight[i]); }, ds);
            copy_1d(out_f32.C_norm_weight, [&](size_t i) { return deq(p.C_norm_weight[i]); }, ds);
            copy_mimo_hd(out_f32.mimo_x, [&](size_t i) { return deq(p.mimo_x[i]); });
            copy_mimo_hd(out_f32.mimo_z, [&](size_t i) { return deq(p.mimo_z[i]); });
            copy_mimo_hd(out_f32.mimo_o, [&](size_t i) { return deq(p.mimo_o[i]); });
            copy_1d(out_f32.D_proj, [&](size_t i) { return deq(p.D_proj[i]); }, nh);
            copy_out_proj([&](size_t i) { return deq(p.out_proj_w[i]); });
            copy_film([&](size_t i) { return deq(p.film_proj_w[i]); });
            for (size_t r = 0; r < 2 * nh; ++r)
                for (size_t c = 0; c < cd; ++c)
                    out_f32.dt_film_proj_w[r][c] = deq(p.dt_film_proj_w[r * cd + c]);
            return 0;
        }
        case QuantFormat::FP16: {
            WeightLayerPointersF16 p;
            if (get_layer_pointers_f16(layer, p) != 0) return -3;

            copy_in_proj([&](size_t r, size_t c) { return fp16_to_fp32(p.in_proj_w[r * dm + c]); });
            copy_1d(out_f32.dt_bias, [&](size_t i) { return fp16_to_fp32(p.dt_bias[i]); }, nh);
            copy_mimo(out_f32.B_bias, [&](size_t i) { return fp16_to_fp32(p.B_bias[i]); });
            copy_mimo(out_f32.C_bias, [&](size_t i) { return fp16_to_fp32(p.C_bias[i]); });
            copy_1d(out_f32.B_norm_weight, [&](size_t i) { return fp16_to_fp32(p.B_norm_weight[i]); }, ds);
            copy_1d(out_f32.C_norm_weight, [&](size_t i) { return fp16_to_fp32(p.C_norm_weight[i]); }, ds);
            copy_mimo_hd(out_f32.mimo_x, [&](size_t i) { return fp16_to_fp32(p.mimo_x[i]); });
            copy_mimo_hd(out_f32.mimo_z, [&](size_t i) { return fp16_to_fp32(p.mimo_z[i]); });
            copy_mimo_hd(out_f32.mimo_o, [&](size_t i) { return fp16_to_fp32(p.mimo_o[i]); });
            copy_1d(out_f32.D_proj, [&](size_t i) { return fp16_to_fp32(p.D_proj[i]); }, nh);
            copy_out_proj([&](size_t i) { return fp16_to_fp32(p.out_proj_w[i]); });
            copy_film([&](size_t i) { return fp16_to_fp32(p.film_proj_w[i]); });
            for (size_t r = 0; r < 2 * nh; ++r)
                for (size_t c = 0; c < cd; ++c)
                    out_f32.dt_film_proj_w[r][c] = fp16_to_fp32(p.dt_film_proj_w[r * cd + c]);
            return 0;
        }
    }
    return -100;
}

} // namespace mamba
