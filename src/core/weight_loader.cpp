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
    const char* env = std::getenv("MAMBA_WEIGHTS");
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

    if (header_->magic != WEIGHT_MAGIC) {
        unload();
        return -11;
    }

    if (header_->version != WEIGHT_VERSION) {
        unload();
        return -12;
    }

    // Validate dimensions against the compiled-in maxima.  The C++ side
    // uses fixed-size SSMWeights, so the file must fit.
    if (header_->d_state   == 0 || header_->d_state   > MAX_D_STATE)  { unload(); return -14; }
    if (header_->d_model   == 0 || header_->d_model   > MAX_D_MODEL)  { unload(); return -14; }
    if (header_->d_inner   == 0 || header_->d_inner   > MAX_D_INNER)  { unload(); return -14; }
    if (header_->d_conv    == 0 || header_->d_conv    > MAX_D_CONV)   { unload(); return -14; }
    if (header_->cond_dim  == 0 || header_->cond_dim  > COND_DIM)     { unload(); return -14; }
    if (header_->num_layers == 0 || header_->num_layers > MAX_LAYERS) { unload(); return -14; }

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

    // Bounds check: weight data must fit in the file
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
                               MAX_D_CONV, MAX_LAYERS);
}

bool WeightLoader::validate_dimensions(uint32_t d_state, uint32_t d_model,
                                        uint32_t d_inner, uint32_t d_conv,
                                        uint32_t num_layers) const {
    if (!header_) return false;
    return header_->d_state    == d_state  &&
           header_->d_model    == d_model  &&
           header_->d_inner    == d_inner  &&
           header_->d_conv     == d_conv   &&
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
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t elems = 0;
    elems += ds;                          // A_log
    elems += ds;                          // D_proj
    elems += ds;                          // dt_bias
    elems += ds * di;                     // dt_proj_w
    elems += di * ds;                     // B_proj_w
    elems += di * ds;                     // C_proj_w
    elems += dc * di;                     // conv1d_w
    elems += di;                          // conv1d_b
    elems += di * dm;                     // W_in
    elems += dm * di;                     // W_out
    elems += dm * ds;                     // W_ssm
    elems += di;                          // b_in
    elems += ds * cd;                     // cond_proj_dt
    elems += ds * cd;                     // cond_proj_B
    elems += ds;                          // cond_dt_bias
    elems += ds;                          // cond_B_bias
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
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t off = 0;
    out.A_log        = fbase + off; off += ds;
    out.D_proj       = fbase + off; off += ds;
    out.dt_bias      = fbase + off; off += ds;
    out.dt_proj_w    = fbase + off; off += ds * di;
    out.B_proj_w     = fbase + off; off += di * ds;
    out.C_proj_w     = fbase + off; off += di * ds;
    out.conv1d_w     = fbase + off; off += dc * di;
    out.conv1d_b     = fbase + off; off += di;
    out.W_in         = fbase + off; off += di * dm;
    out.W_out        = fbase + off; off += dm * di;
    out.W_ssm        = fbase + off; off += dm * ds;
    out.b_in         = fbase + off; off += di;
    out.cond_proj_dt = fbase + off; off += ds * cd;
    out.cond_proj_B  = fbase + off; off += ds * cd;
    out.cond_dt_bias = fbase + off; off += ds;
    out.cond_B_bias  = fbase + off;
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
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t off = 0;
    out.A_log        = ibase + off; off += ds;
    out.D_proj       = ibase + off; off += ds;
    out.dt_bias      = ibase + off; off += ds;
    out.dt_proj_w    = ibase + off; off += ds * di;
    out.B_proj_w     = ibase + off; off += di * ds;
    out.C_proj_w     = ibase + off; off += di * ds;
    out.conv1d_w     = ibase + off; off += dc * di;
    out.conv1d_b     = ibase + off; off += di;
    out.W_in         = ibase + off; off += di * dm;
    out.W_out        = ibase + off; off += dm * di;
    out.W_ssm        = ibase + off; off += dm * ds;
    out.b_in         = ibase + off; off += di;
    out.cond_proj_dt = ibase + off; off += ds * cd;
    out.cond_proj_B  = ibase + off; off += ds * cd;
    out.cond_dt_bias = ibase + off; off += ds;
    out.cond_B_bias  = ibase + off;
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
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t off = 0;
    out.A_log        = hbase + off; off += ds;
    out.D_proj       = hbase + off; off += ds;
    out.dt_bias      = hbase + off; off += ds;
    out.dt_proj_w    = hbase + off; off += ds * di;
    out.B_proj_w     = hbase + off; off += di * ds;
    out.C_proj_w     = hbase + off; off += di * ds;
    out.conv1d_w     = hbase + off; off += dc * di;
    out.conv1d_b     = hbase + off; off += di;
    out.W_in         = hbase + off; off += di * dm;
    out.W_out        = hbase + off; off += dm * di;
    out.W_ssm        = hbase + off; off += dm * ds;
    out.b_in         = hbase + off; off += di;
    out.cond_proj_dt = hbase + off; off += ds * cd;
    out.cond_proj_B  = hbase + off; off += ds * cd;
    out.cond_dt_bias = hbase + off; off += ds;
    out.cond_B_bias  = hbase + off;
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
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    std::memset(&out_f32, 0, sizeof(out_f32));

    auto dequant = [](const int8_t* src, float* dst, size_t n,
                      float scale, int32_t zero_point) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = scale * (static_cast<float>(src[i]) -
                              static_cast<float>(zero_point));
        }
    };
    auto dequant_f16 = [](const uint16_t* src, float* dst, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            dst[i] = fp16_to_fp32(src[i]);
        }
    };
    auto copy_f32 = [](const float* src, float* dst, size_t n) {
        std::memcpy(dst, src, n * sizeof(float));
    };

    switch (quant_format()) {
        case QuantFormat::Float32: {
            WeightLayerPointers p;
            if (get_layer_pointers(layer, p) != 0) return -3;
            copy_f32(p.A_log,        out_f32.A_log,        ds);
            copy_f32(p.D_proj,       out_f32.D_proj,       ds);
            copy_f32(p.dt_bias,      out_f32.dt_bias,      ds);
            copy_f32(p.dt_proj_w,    out_f32.dt_proj_w[0], ds * di);
            copy_f32(p.B_proj_w,     out_f32.B_proj_w[0],  di * ds);
            copy_f32(p.C_proj_w,     out_f32.C_proj_w[0],  di * ds);
            copy_f32(p.conv1d_w,     out_f32.conv1d_w[0],  dc * di);
            copy_f32(p.conv1d_b,     out_f32.conv1d_b,     di);
            copy_f32(p.W_in,         out_f32.W_in[0],      di * dm);
            copy_f32(p.W_out,        out_f32.W_out[0],     dm * di);
            copy_f32(p.W_ssm,        out_f32.W_ssm[0],     dm * ds);
            copy_f32(p.b_in,         out_f32.b_in,         di);
            copy_f32(p.cond_proj_dt, out_f32.cond_proj_dt[0], ds * cd);
            copy_f32(p.cond_proj_B,  out_f32.cond_proj_B[0],  ds * cd);
            copy_f32(p.cond_dt_bias, out_f32.cond_dt_bias,    ds);
            copy_f32(p.cond_B_bias,  out_f32.cond_B_bias,     ds);
            return 0;
        }
        case QuantFormat::Int8Tensor: {
            WeightLayerPointersI8 p;
            if (get_layer_pointers_i8(layer, p) != 0) return -3;
            TensorQParam qp;
            if (get_tensor_qparam(qp) != 0) return -4;
            dequant(p.A_log,        out_f32.A_log,        ds,        qp.scale, qp.zero_point);
            dequant(p.D_proj,       out_f32.D_proj,       ds,        qp.scale, qp.zero_point);
            dequant(p.dt_bias,      out_f32.dt_bias,      ds,        qp.scale, qp.zero_point);
            dequant(p.dt_proj_w,    out_f32.dt_proj_w[0], ds * di,   qp.scale, qp.zero_point);
            dequant(p.B_proj_w,     out_f32.B_proj_w[0],  di * ds,   qp.scale, qp.zero_point);
            dequant(p.C_proj_w,     out_f32.C_proj_w[0],  di * ds,   qp.scale, qp.zero_point);
            dequant(p.conv1d_w,     out_f32.conv1d_w[0],  dc * di,   qp.scale, qp.zero_point);
            dequant(p.conv1d_b,     out_f32.conv1d_b,     di,        qp.scale, qp.zero_point);
            dequant(p.W_in,         out_f32.W_in[0],      di * dm,   qp.scale, qp.zero_point);
            dequant(p.W_out,        out_f32.W_out[0],     dm * di,   qp.scale, qp.zero_point);
            dequant(p.W_ssm,        out_f32.W_ssm[0],     dm * ds,   qp.scale, qp.zero_point);
            dequant(p.b_in,         out_f32.b_in,         di,        qp.scale, qp.zero_point);
            dequant(p.cond_proj_dt, out_f32.cond_proj_dt[0], ds * cd, qp.scale, qp.zero_point);
            dequant(p.cond_proj_B,  out_f32.cond_proj_B[0],  ds * cd, qp.scale, qp.zero_point);
            dequant(p.cond_dt_bias, out_f32.cond_dt_bias,    ds,     qp.scale, qp.zero_point);
            dequant(p.cond_B_bias,  out_f32.cond_B_bias,     ds,     qp.scale, qp.zero_point);
            return 0;
        }
        case QuantFormat::Int8PerLayer: {
            WeightLayerPointersI8 p;
            if (get_layer_pointers_i8(layer, p) != 0) return -3;
            LayerQParam qp;
            if (get_layer_qparam(layer, qp) != 0) return -4;
            dequant(p.A_log,        out_f32.A_log,        ds,        qp.scale, qp.zero_point);
            dequant(p.D_proj,       out_f32.D_proj,       ds,        qp.scale, qp.zero_point);
            dequant(p.dt_bias,      out_f32.dt_bias,      ds,        qp.scale, qp.zero_point);
            dequant(p.dt_proj_w,    out_f32.dt_proj_w[0], ds * di,   qp.scale, qp.zero_point);
            dequant(p.B_proj_w,     out_f32.B_proj_w[0],  di * ds,   qp.scale, qp.zero_point);
            dequant(p.C_proj_w,     out_f32.C_proj_w[0],  di * ds,   qp.scale, qp.zero_point);
            dequant(p.conv1d_w,     out_f32.conv1d_w[0],  dc * di,   qp.scale, qp.zero_point);
            dequant(p.conv1d_b,     out_f32.conv1d_b,     di,        qp.scale, qp.zero_point);
            dequant(p.W_in,         out_f32.W_in[0],      di * dm,   qp.scale, qp.zero_point);
            dequant(p.W_out,        out_f32.W_out[0],     dm * di,   qp.scale, qp.zero_point);
            dequant(p.W_ssm,        out_f32.W_ssm[0],     dm * ds,   qp.scale, qp.zero_point);
            dequant(p.b_in,         out_f32.b_in,         di,        qp.scale, qp.zero_point);
            dequant(p.cond_proj_dt, out_f32.cond_proj_dt[0], ds * cd, qp.scale, qp.zero_point);
            dequant(p.cond_proj_B,  out_f32.cond_proj_B[0],  ds * cd, qp.scale, qp.zero_point);
            dequant(p.cond_dt_bias, out_f32.cond_dt_bias,    ds,     qp.scale, qp.zero_point);
            dequant(p.cond_B_bias,  out_f32.cond_B_bias,     ds,     qp.scale, qp.zero_point);
            return 0;
        }
        case QuantFormat::FP16: {
            WeightLayerPointersF16 p;
            if (get_layer_pointers_f16(layer, p) != 0) return -3;
            dequant_f16(p.A_log,        out_f32.A_log,        ds);
            dequant_f16(p.D_proj,       out_f32.D_proj,       ds);
            dequant_f16(p.dt_bias,      out_f32.dt_bias,      ds);
            dequant_f16(p.dt_proj_w,    out_f32.dt_proj_w[0], ds * di);
            dequant_f16(p.B_proj_w,     out_f32.B_proj_w[0],  di * ds);
            dequant_f16(p.C_proj_w,     out_f32.C_proj_w[0],  di * ds);
            dequant_f16(p.conv1d_w,     out_f32.conv1d_w[0],  dc * di);
            dequant_f16(p.conv1d_b,     out_f32.conv1d_b,     di);
            dequant_f16(p.W_in,         out_f32.W_in[0],      di * dm);
            dequant_f16(p.W_out,        out_f32.W_out[0],     dm * di);
            dequant_f16(p.W_ssm,        out_f32.W_ssm[0],     dm * ds);
            dequant_f16(p.b_in,         out_f32.b_in,         di);
            dequant_f16(p.cond_proj_dt, out_f32.cond_proj_dt[0], ds * cd);
            dequant_f16(p.cond_proj_B,  out_f32.cond_proj_B[0],  ds * cd);
            dequant_f16(p.cond_dt_bias, out_f32.cond_dt_bias,    ds);
            dequant_f16(p.cond_B_bias,  out_f32.cond_B_bias,     ds);
            return 0;
        }
    }
    return -100;
}

} // namespace mamba
