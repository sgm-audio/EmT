#include "core/weight_loader.h"
#include <cstring>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#endif

namespace mamba {

WeightLoader::WeightLoader()
    : mapped_data_(nullptr), mapped_size_(0), fd_(-1),
      header_(nullptr), weight_data_(nullptr), weight_data_floats_(0) {}

WeightLoader::~WeightLoader() {
    unload();
}

int WeightLoader::load(const char* path) {
    unload();

#ifdef _WIN32
    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_READONLY, nullptr);
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
    fd_ = open(path, O_RDONLY);
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

    weight_data_ = reinterpret_cast<const float*>(
        static_cast<const uint8_t*>(mapped_data_) + sizeof(WeightFileHeader));
    weight_data_floats_ = static_cast<size_t>(header_->weight_data_size) / sizeof(float);

    size_t expected = floats_per_layer() * header_->num_layers;
    if (weight_data_floats_ < expected) {
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
    weight_data_floats_ = 0;
}

bool WeightLoader::validate_dimensions(uint32_t d_state, uint32_t d_model,
                                        uint32_t d_inner, uint32_t d_conv,
                                        uint32_t num_layers) const {
    if (!header_) return false;
    return header_->d_state == d_state &&
           header_->d_model == d_model &&
           header_->d_inner == d_inner &&
           header_->d_conv == d_conv &&
           header_->num_layers == num_layers;
}

size_t WeightLoader::floats_per_layer() const {
    if (!header_) return 0;
    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t total = 0;
    total += ds;                        // A_log
    total += ds;                        // D_proj
    total += ds;                        // dt_bias
    total += ds * di;                   // dt_proj_w
    total += di * ds;                   // B_proj_w
    total += di * ds;                   // C_proj_w
    total += dc * di;                   // conv1d_w
    total += di;                        // conv1d_b
    total += di * dm;                   // W_in
    total += dm * di;                   // W_out
    total += dm * ds;                   // W_ssm
    total += di;                        // b_in
    total += ds * cd;                   // cond_proj_dt
    total += ds * cd;                   // cond_proj_B

    return total;
}

int WeightLoader::get_layer_pointers(uint32_t layer, WeightLayerPointers& out) const {
    if (!header_ || !weight_data_) return -1;
    if (layer >= header_->num_layers) return -2;

    uint32_t ds = header_->d_state;
    uint32_t dm = header_->d_model;
    uint32_t di = header_->d_inner;
    uint32_t dc = header_->d_conv;
    uint32_t cd = header_->cond_dim;

    size_t fpl = floats_per_layer();
    const float* base = weight_data_ + layer * fpl;

    size_t offset = 0;

    out.A_log = base + offset; offset += ds;
    out.D_proj = base + offset; offset += ds;
    out.dt_bias = base + offset; offset += ds;
    out.dt_proj_w = base + offset; offset += ds * di;
    out.B_proj_w = base + offset; offset += di * ds;
    out.C_proj_w = base + offset; offset += di * ds;
    out.conv1d_w = base + offset; offset += dc * di;
    out.conv1d_b = base + offset; offset += di;
    out.W_in = base + offset; offset += di * dm;
    out.W_out = base + offset; offset += dm * di;
    out.W_ssm = base + offset; offset += dm * ds;
    out.b_in = base + offset; offset += di;
    out.cond_proj_dt = base + offset; offset += ds * cd;
    out.cond_proj_B = base + offset; offset += ds * cd;

    return 0;
}

} // namespace mamba
