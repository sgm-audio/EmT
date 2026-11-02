#include <gtest/gtest.h>
#include "core/weight_loader.h"
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>
#include <filesystem>

#if defined(_WIN32)
#include <windows.h>
#include <io.h>
static int mamba_getpid() { return static_cast<int>(GetCurrentProcessId()); }
#define mamba_unlink _unlink
#else
#include <unistd.h>
static int mamba_getpid() { return static_cast<int>(getpid()); }
#define mamba_unlink unlink
#endif

using namespace mamba;

class WeightLoaderTest : public ::testing::Test {
protected:
    char tmp_path_[512];

    void SetUp() override {
        std::filesystem::path p = std::filesystem::temp_directory_path() /
            ("mamba_test_weights_" + std::to_string(mamba_getpid()) + ".bin");
        std::strncpy(tmp_path_, p.string().c_str(), sizeof(tmp_path_) - 1);
        tmp_path_[sizeof(tmp_path_) - 1] = '\0';
    }

    void TearDown() override {
        mamba_unlink(tmp_path_);
    }

    size_t floats_per_layer() const {
        uint32_t ds = MAX_D_STATE;
        uint32_t dm = MAX_D_MODEL;
        uint32_t di = MAX_D_INNER;
        uint32_t dc = MAX_D_CONV;
        uint32_t cd = COND_DIM;
        return ds + ds + ds +
               ds * di + di * ds + di * ds +
               dc * di + di +
               di * dm + dm * di + di +
               dm * ds +
               ds * cd + ds * cd +
               ds + ds;     // cond_dt_bias, cond_B_bias
    }

    bool create_valid_weight_file(uint32_t num_layers = 1) {
        size_t fpl = floats_per_layer();
        size_t total_floats = fpl * num_layers;
        size_t data_size = total_floats * sizeof(float);

        FILE* f = fopen(tmp_path_, "wb");
        if (!f) return false;

        WeightFileHeader hdr{};
        hdr.magic = WEIGHT_MAGIC;
        hdr.version = WEIGHT_VERSION;
        hdr.d_state = MAX_D_STATE;
        hdr.d_model = MAX_D_MODEL;
        hdr.d_inner = MAX_D_INNER;
        hdr.d_conv = MAX_D_CONV;
        hdr.num_layers = num_layers;
        hdr.expansion_factor = MAX_D_INNER / MAX_D_MODEL;
        hdr.cond_dim = COND_DIM;
        hdr.weight_data_size = static_cast<uint64_t>(data_size);

        fwrite(&hdr, sizeof(hdr), 1, f);

        std::vector<float> data(total_floats, 0.0f);
        size_t offset = 0;

        for (uint32_t layer = 0; layer < num_layers; ++layer) {
            for (size_t i = 0; i < MAX_D_STATE; ++i) data[offset++] = -1.0f;
            for (size_t i = 0; i < MAX_D_STATE; ++i) data[offset++] = 0.5f;
            for (size_t i = 0; i < MAX_D_STATE; ++i) data[offset++] = 0.5f;
            for (size_t i = 0; i < MAX_D_STATE * MAX_D_INNER; ++i) data[offset++] = 0.1f;
            for (size_t i = 0; i < MAX_D_INNER * MAX_D_STATE; ++i) data[offset++] = 0.1f;
            for (size_t i = 0; i < MAX_D_INNER * MAX_D_STATE; ++i) data[offset++] = 0.1f;
            for (size_t i = 0; i < MAX_D_CONV * MAX_D_INNER; ++i) data[offset++] = 0.05f;
            for (size_t i = 0; i < MAX_D_INNER; ++i) data[offset++] = 0.1f;
            for (size_t i = 0; i < MAX_D_INNER * MAX_D_MODEL; ++i) data[offset++] = 0.05f;
            for (size_t i = 0; i < MAX_D_MODEL * MAX_D_INNER; ++i) data[offset++] = 0.01f;
            for (size_t i = 0; i < MAX_D_MODEL * MAX_D_STATE; ++i) {
                data[offset++] = (i % (MAX_D_STATE + 1) == 0) ? 1.0f : 0.0f;
            }
            for (size_t i = 0; i < MAX_D_INNER; ++i) data[offset++] = 0.01f;
            for (size_t i = 0; i < MAX_D_STATE * COND_DIM; ++i) data[offset++] = 0.0f;
            for (size_t i = 0; i < MAX_D_STATE * COND_DIM; ++i) data[offset++] = 0.0f;
            for (size_t i = 0; i < MAX_D_STATE; ++i) data[offset++] = 0.0f;  // cond_dt_bias
            for (size_t i = 0; i < MAX_D_STATE; ++i) data[offset++] = 0.0f;  // cond_B_bias
        }

        fwrite(data.data(), sizeof(float), total_floats, f);
        fclose(f);
        return true;
    }
};

TEST_F(WeightLoaderTest, LoadValidFile) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    int rc = loader.load(tmp_path_);
    EXPECT_EQ(rc, 0);
    EXPECT_TRUE(loader.is_loaded());
    EXPECT_EQ(loader.get_num_layers(), 1u);

    loader.unload();
    EXPECT_FALSE(loader.is_loaded());
}

TEST_F(WeightLoaderTest, LoadNonexistentFile) {
    WeightLoader loader;
    int rc = loader.load("/tmp/mamba_nonexistent_weights.bin");
    EXPECT_NE(rc, 0);
    EXPECT_FALSE(loader.is_loaded());
}

TEST_F(WeightLoaderTest, ValidateDimensions) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path_), 0);

    EXPECT_TRUE(loader.validate_dimensions(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1));
    EXPECT_FALSE(loader.validate_dimensions(MAX_D_STATE + 1, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 1));

    loader.unload();
}

TEST_F(WeightLoaderTest, GetLayerPointers) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path_), 0);

    WeightLayerPointers ptrs;
    int rc = loader.get_layer_pointers(0, ptrs);
    EXPECT_EQ(rc, 0);
    EXPECT_NE(ptrs.A_log, nullptr);
    EXPECT_NE(ptrs.D_proj, nullptr);
    EXPECT_NE(ptrs.dt_bias, nullptr);
    EXPECT_NE(ptrs.dt_proj_w, nullptr);
    EXPECT_NE(ptrs.B_proj_w, nullptr);
    EXPECT_NE(ptrs.C_proj_w, nullptr);
    EXPECT_NE(ptrs.conv1d_w, nullptr);
    EXPECT_NE(ptrs.conv1d_b, nullptr);
    EXPECT_NE(ptrs.W_in, nullptr);
    EXPECT_NE(ptrs.W_out, nullptr);
    EXPECT_NE(ptrs.W_ssm, nullptr);
    EXPECT_NE(ptrs.b_in, nullptr);

    EXPECT_FLOAT_EQ(ptrs.A_log[0], -1.0f);
    EXPECT_FLOAT_EQ(ptrs.D_proj[0], 0.5f);
    EXPECT_FLOAT_EQ(ptrs.dt_bias[0], 0.5f);

    rc = loader.get_layer_pointers(1, ptrs);
    EXPECT_NE(rc, 0);

    loader.unload();
}

TEST_F(WeightLoaderTest, LoadMultiLayer) {
    const uint32_t kNumLayers = 2;  // must fit in MAX_LAYERS

    ASSERT_TRUE(create_valid_weight_file(kNumLayers));

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path_), 0);
    EXPECT_EQ(loader.get_num_layers(), kNumLayers);

    WeightLayerPointers ptrs;
    for (uint32_t i = 0; i < kNumLayers; ++i) {
        EXPECT_EQ(loader.get_layer_pointers(i, ptrs), 0);
    }
    EXPECT_NE(loader.get_layer_pointers(kNumLayers, ptrs), 0);

    loader.unload();
}

TEST_F(WeightLoaderTest, BadMagic) {
    FILE* f = fopen(tmp_path_, "wb");
    ASSERT_NE(f, nullptr);
    WeightFileHeader hdr{};
    hdr.magic = 0xDEADBEEF;
    hdr.version = WEIGHT_VERSION;
    hdr.num_layers = 1;
    hdr.weight_data_size = 0;
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_path_), -11);
}

TEST_F(WeightLoaderTest, BadVersion) {
    FILE* f = fopen(tmp_path_, "wb");
    ASSERT_NE(f, nullptr);
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = 999;
    hdr.num_layers = 1;
    hdr.weight_data_size = 0;
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_path_), -12);
}

TEST_F(WeightLoaderTest, FileTooSmall) {
    FILE* f = fopen(tmp_path_, "wb");
    ASSERT_NE(f, nullptr);
    uint32_t bad = 0;
    fwrite(&bad, sizeof(bad), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_path_), -10);
}

TEST(WeightIntegration, LoadWeightsThenProcessNonZero) {
    char tmp_path[512];
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        ("mamba_integration_" + std::to_string(mamba_getpid()) + ".bin");
    std::strncpy(tmp_path, p.string().c_str(), sizeof(tmp_path) - 1);
    tmp_path[sizeof(tmp_path) - 1] = '\0';

    size_t fpl = MAX_D_STATE + MAX_D_STATE + MAX_D_STATE +
                 MAX_D_STATE * MAX_D_INNER + MAX_D_INNER * MAX_D_STATE + MAX_D_INNER * MAX_D_STATE +
                 MAX_D_CONV * MAX_D_INNER + MAX_D_INNER +
                 MAX_D_INNER * MAX_D_MODEL + MAX_D_MODEL * MAX_D_INNER + MAX_D_INNER +
                 MAX_D_MODEL * MAX_D_STATE +
                 MAX_D_STATE * COND_DIM + MAX_D_STATE * COND_DIM +
                 MAX_D_STATE + MAX_D_STATE;   // cond_dt_bias, cond_B_bias
    size_t total_floats = fpl * 2;
    size_t data_size = total_floats * sizeof(float);

    FILE* f = fopen(tmp_path, "wb");
    ASSERT_NE(f, nullptr);
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = MAX_D_STATE;
    hdr.d_model = MAX_D_MODEL;
    hdr.d_inner = MAX_D_INNER;
    hdr.d_conv = MAX_D_CONV;
    hdr.num_layers = 2;
    hdr.expansion_factor = MAX_D_INNER / MAX_D_MODEL;
    hdr.cond_dim = COND_DIM;
    hdr.weight_data_size = static_cast<uint64_t>(data_size);
    fwrite(&hdr, sizeof(hdr), 1, f);

    std::vector<float> wdata(total_floats, 0.0f);
    size_t off = 0;
    for (uint32_t layer = 0; layer < 2; ++layer) {
        for (size_t i = 0; i < MAX_D_STATE; ++i) wdata[off++] = -1.0f;
        for (size_t i = 0; i < MAX_D_STATE; ++i) wdata[off++] = 0.5f;
        for (size_t i = 0; i < MAX_D_STATE; ++i) wdata[off++] = 0.5f;
        for (size_t i = 0; i < MAX_D_STATE * MAX_D_INNER; ++i) wdata[off++] = 0.1f;
        for (size_t i = 0; i < MAX_D_INNER * MAX_D_STATE; ++i) wdata[off++] = 0.1f;
        for (size_t i = 0; i < MAX_D_INNER * MAX_D_STATE; ++i) wdata[off++] = 0.1f;
        for (size_t i = 0; i < MAX_D_CONV * MAX_D_INNER; ++i) wdata[off++] = 0.05f;
        for (size_t i = 0; i < MAX_D_INNER; ++i) wdata[off++] = 0.1f;
        for (size_t i = 0; i < MAX_D_INNER * MAX_D_MODEL; ++i) wdata[off++] = 0.05f;
        for (size_t i = 0; i < MAX_D_MODEL * MAX_D_INNER; ++i) wdata[off++] = 0.01f;
        for (size_t i = 0; i < MAX_D_MODEL * MAX_D_STATE; ++i) {
            wdata[off++] = (i % (MAX_D_STATE + 1) == 0) ? 1.0f : 0.0f;
        }
        for (size_t i = 0; i < MAX_D_INNER; ++i) wdata[off++] = 0.01f;
        for (size_t i = 0; i < MAX_D_STATE * COND_DIM; ++i) wdata[off++] = 0.0f;
        for (size_t i = 0; i < MAX_D_STATE * COND_DIM; ++i) wdata[off++] = 0.0f;
        for (size_t i = 0; i < MAX_D_STATE; ++i) wdata[off++] = 0.0f;  // cond_dt_bias
        for (size_t i = 0; i < MAX_D_STATE; ++i) wdata[off++] = 0.0f;  // cond_B_bias
    }
    fwrite(wdata.data(), sizeof(float), total_floats, f);
    fclose(f);

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path), 0);
    ASSERT_TRUE(loader.validate_dimensions(MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 2));

    MambaConfig cfg{MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_D_CONV, 2, COND_DIM};
    SSMInferenceCore core(cfg);

    for (uint32_t layer = 0; layer < 2; ++layer) {
        WeightLayerPointers ptrs;
        ASSERT_EQ(loader.get_layer_pointers(layer, ptrs), 0);

        SSMWeights w{};
        std::memcpy(w.A_log, ptrs.A_log, MAX_D_STATE * sizeof(float));
        std::memcpy(w.D_proj, ptrs.D_proj, MAX_D_STATE * sizeof(float));
        std::memcpy(w.dt_bias, ptrs.dt_bias, MAX_D_STATE * sizeof(float));
        std::memcpy(w.dt_proj_w, ptrs.dt_proj_w, MAX_D_STATE * MAX_D_INNER * sizeof(float));
        std::memcpy(w.B_proj_w, ptrs.B_proj_w, MAX_D_INNER * MAX_D_STATE * sizeof(float));
        std::memcpy(w.C_proj_w, ptrs.C_proj_w, MAX_D_INNER * MAX_D_STATE * sizeof(float));
        std::memcpy(w.conv1d_w, ptrs.conv1d_w, MAX_D_CONV * MAX_D_INNER * sizeof(float));
        std::memcpy(w.conv1d_b, ptrs.conv1d_b, MAX_D_INNER * sizeof(float));
        std::memcpy(w.W_in, ptrs.W_in, MAX_D_INNER * MAX_D_MODEL * sizeof(float));
        std::memcpy(w.W_out, ptrs.W_out, MAX_D_MODEL * MAX_D_INNER * sizeof(float));
        std::memcpy(w.W_ssm, ptrs.W_ssm, MAX_D_MODEL * MAX_D_STATE * sizeof(float));
        std::memcpy(w.b_in, ptrs.b_in, MAX_D_INNER * sizeof(float));

        core.set_layer_weights(layer, w);
    }

    float out = core.process_sample(1.0f);
    EXPECT_NE(out, 0.0f);
    EXPECT_FALSE(std::isnan(out));
    EXPECT_FALSE(std::isinf(out));

    float out2 = core.process_sample(0.5f);
    EXPECT_FALSE(std::isnan(out2));
    EXPECT_FALSE(std::isinf(out2));

    loader.unload();
    mamba_unlink(tmp_path);
}

// ---------------------------------------------------------------------------
// FP16 -> FP32 IEEE 754 conversion (header-only, see weight_loader.h)
// ---------------------------------------------------------------------------

TEST(FP16, ZeroAndSignedZero) {
    EXPECT_EQ(fp16_to_fp32(0x0000), 0.0f);
    EXPECT_EQ(fp16_to_fp32(0x8000), -0.0f);
}

TEST(FP16, OneAndMinusOne) {
    EXPECT_EQ(fp16_to_fp32(0x3C00), 1.0f);   // sign=0, exp=15, mant=0
    EXPECT_EQ(fp16_to_fp32(0xBC00), -1.0f);  // sign=1, exp=15, mant=0
}

TEST(FP16, TwoAndHalf) {
    EXPECT_EQ(fp16_to_fp32(0x4000), 2.0f);  // 1.0 * 2^1
    EXPECT_EQ(fp16_to_fp32(0x3800), 0.5f);  // 1.0 * 2^-1
}

TEST(FP16, InfAndNaN) {
    EXPECT_EQ(fp16_to_fp32(0x7C00),  std::numeric_limits<float>::infinity());
    EXPECT_EQ(fp16_to_fp32(0xFC00), -std::numeric_limits<float>::infinity());

    float nan_val = fp16_to_fp32(0x7E00);  // NaN
    EXPECT_TRUE(std::isnan(nan_val));
}

TEST(FP16, RoundTripKnownValues) {
    // Test against a known LUT for a few hundred values.
    // Each value: write to fp16, convert back, check within 1e-3.
    for (uint32_t bits = 0; bits < 0x10000; bits += 0x101) {
        uint16_t h = static_cast<uint16_t>(bits);
        float f = fp16_to_fp32(h);
        if (std::isnan(f) || std::isinf(f)) continue;
        // FTZ flushes subnormals to zero; skip subnormals.
        uint32_t exp_field = (bits >> 10) & 0x1F;
        if (exp_field == 0) continue;
        // The conversion is exact for any value representable in FP16
        // (the result is exactly the same as writing bits to a float).
        uint32_t mant = bits & 0x3FF;
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t fp32_bits = (sign << 31) | (((exp_field - 15 + 127) & 0xFF) << 23) | (mant << 13);
        float expected;
        std::memcpy(&expected, &fp32_bits, sizeof(expected));
        EXPECT_EQ(f, expected) << "Mismatch at bits=0x" << std::hex << bits;
    }
}

// ---------------------------------------------------------------------------
// INT8 quantization round-trip
// ---------------------------------------------------------------------------

namespace {

struct QFile {
    char path[512];
    ~QFile() { mamba_unlink(path); }
};

QFile make_i8_per_layer_file(float scale, int32_t zp) {
    char tmp[512];
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        ("mamba_q8_" + std::to_string(mamba_getpid()) + ".bin");
    std::strncpy(tmp, p.string().c_str(), sizeof(tmp) - 1);
    tmp[sizeof(tmp) - 1] = '\0';

    uint32_t ds = MAX_D_STATE, dm = MAX_D_MODEL, di = MAX_D_INNER, dc = MAX_D_CONV;
    uint32_t cd = COND_DIM;
    size_t elems_per_layer = 3 * ds + 3 * ds * di + dc * di + 2 * di +
                             2 * di * dm + dm * ds + 2 * ds * cd +
                             2 * ds;     // cond_dt_bias, cond_B_bias
    size_t bytes_per_layer = elems_per_layer;  // INT8 = 1 byte
    size_t q_meta_size = sizeof(float) + sizeof(int32_t) + 8;  // 16 bytes per layer
    size_t weight_data_size = bytes_per_layer;

    std::FILE* f = std::fopen(tmp, "wb");
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = ds; hdr.d_model = dm; hdr.d_inner = di; hdr.d_conv = dc;
    hdr.num_layers = 1;
    hdr.expansion_factor = di / dm;
    hdr.cond_dim = cd;
    hdr.quantization = static_cast<uint32_t>(QuantFormat::Int8PerLayer);
    hdr.q_metadata_size = static_cast<uint32_t>(q_meta_size);
    hdr.weight_data_size = static_cast<uint64_t>(weight_data_size);
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    // Write per-layer q-metadata
    float s = scale;
    int32_t z = zp;
    uint8_t pad[8] = {0};
    std::fwrite(&s, sizeof(s), 1, f);
    std::fwrite(&z, sizeof(z), 1, f);
    std::fwrite(pad, 1, 8, f);

    // Write INT8 data: 0 -> zp, 1 -> zp+1, etc. (small values)
    std::vector<int8_t> data(weight_data_size);
    for (size_t i = 0; i < data.size(); ++i) {
        int32_t v = static_cast<int32_t>(i % 5) + zp;
        if (v < -128) v = -128;
        if (v > 127) v = 127;
        data[i] = static_cast<int8_t>(v);
    }
    std::fwrite(data.data(), 1, weight_data_size, f);
    std::fclose(f);

    QFile result;
    std::strncpy(result.path, tmp, sizeof(result.path) - 1);
    result.path[sizeof(result.path) - 1] = '\0';
    return result;
}

} // namespace

TEST(WeightQuantization, Int8PerLayerRoundTrip) {
    QFile qf = make_i8_per_layer_file(0.01f, 0);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), 0);
    ASSERT_EQ(loader.quant_format(), QuantFormat::Int8PerLayer);

    LayerQParam qp;
    ASSERT_EQ(loader.get_layer_qparam(0, qp), 0);
    EXPECT_FLOAT_EQ(qp.scale, 0.01f);
    EXPECT_EQ(qp.zero_point, 0);

    SSMWeights w{};
    ASSERT_EQ(loader.dequantize_layer_to(0, w), 0);

    // First A_log element is data[0] = 0 (zero_point) → 0.01 * (0 - 0) = 0
    EXPECT_NEAR(w.A_log[0], 0.0f, 1e-6f);
    // First A_log element is data[1] = 1 → 0.01 * (1 - 0) = 0.01
    EXPECT_NEAR(w.A_log[1], 0.01f, 1e-6f);

    loader.unload();
}

TEST(WeightQuantization, Int8PerLayerOffsetAndScale) {
    QFile qf = make_i8_per_layer_file(0.5f, 10);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), 0);

    SSMWeights w{};
    ASSERT_EQ(loader.dequantize_layer_to(0, w), 0);
    // data[0] = 10 (zero_point) → 0.5 * (10 - 10) = 0
    EXPECT_NEAR(w.A_log[0], 0.0f, 1e-6f);
    // data[1] = 11 → 0.5 * (11 - 10) = 0.5
    EXPECT_NEAR(w.A_log[1], 0.5f, 1e-6f);
    loader.unload();
}

TEST(WeightQuantization, RejectsWrongFormatAccess) {
    QFile qf = make_i8_per_layer_file(0.01f, 0);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), 0);

    WeightLayerPointers fp;
    EXPECT_NE(loader.get_layer_pointers(0, fp), 0);

    WeightLayerPointersF16 f16p;
    EXPECT_NE(loader.get_layer_pointers_f16(0, f16p), 0);

    TensorQParam tqp;
    EXPECT_NE(loader.get_tensor_qparam(tqp), 0);

    loader.unload();
}

// ---------------------------------------------------------------------------
// Path resolution (env var, default, explicit)
// ---------------------------------------------------------------------------

TEST(WeightLoaderPath, MAMBAWeightsEnvVar) {
    char tmp_path[512];
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        ("mamba_env_" + std::to_string(mamba_getpid()) + ".bin");
    std::strncpy(tmp_path, p.string().c_str(), sizeof(tmp_path) - 1);
    tmp_path[sizeof(tmp_path) - 1] = '\0';

    // Create a valid 1-layer file
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = 1; hdr.d_model = 1; hdr.d_inner = 1; hdr.d_conv = 1;
    hdr.num_layers = 1;
    hdr.expansion_factor = 1;
    hdr.cond_dim = 1;
    size_t fpl = 3 + 3 + 1 + 2 + 2 + 1 + 2 + 2;  // minimal 16-weight for d=1
    hdr.weight_data_size = fpl * sizeof(float);
    std::FILE* f = std::fopen(tmp_path, "wb");
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::vector<float> data(fpl, 0.5f);
    std::fwrite(data.data(), sizeof(float), fpl, f);
    std::fclose(f);

    // Set env var
#ifdef _WIN32
    _putenv_s("MAMBA_WEIGHTS", tmp_path);
#else
    ::setenv("MAMBA_WEIGHTS", tmp_path, 1);
#endif

    WeightLoader loader;
    int rc = loader.load();  // no path arg
    EXPECT_EQ(rc, 0);
    EXPECT_STREQ(loader.source_path(), tmp_path);
    loader.unload();

#ifdef _WIN32
    _putenv_s("MAMBA_WEIGHTS", "");
#else
    ::unsetenv("MAMBA_WEIGHTS");
#endif
    mamba_unlink(tmp_path);
}

TEST(WeightLoaderPath, ExplicitPathOverridesEnv) {
    char env_path[512], good_path[512];
    std::filesystem::path ep = std::filesystem::temp_directory_path() /
        ("mamba_envX_" + std::to_string(mamba_getpid()) + ".bin");
    std::filesystem::path gp = std::filesystem::temp_directory_path() /
        ("mamba_good_" + std::to_string(mamba_getpid()) + ".bin");
    std::strncpy(env_path, ep.string().c_str(), sizeof(env_path) - 1);
    std::strncpy(good_path, gp.string().c_str(), sizeof(good_path) - 1);

    // Write good_path with valid minimal content
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = 1; hdr.d_model = 1; hdr.d_inner = 1; hdr.d_conv = 1;
    hdr.num_layers = 1;
    hdr.expansion_factor = 1;
    hdr.cond_dim = 1;
    size_t fpl = 3 + 3 + 1 + 2 + 2 + 1 + 2 + 2;
    hdr.weight_data_size = fpl * sizeof(float);
    std::FILE* f = std::fopen(good_path, "wb");
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::vector<float> data(fpl, 0.5f);
    std::fwrite(data.data(), sizeof(float), fpl, f);
    std::fclose(f);

    // env_path doesn't exist
#ifdef _WIN32
    _putenv_s("MAMBA_WEIGHTS", env_path);
#else
    ::setenv("MAMBA_WEIGHTS", env_path, 1);
#endif

    WeightLoader loader;
    int rc = loader.load(good_path);  // explicit arg should win
    EXPECT_EQ(rc, 0);
    EXPECT_STREQ(loader.source_path(), good_path);
    loader.unload();

#ifdef _WIN32
    _putenv_s("MAMBA_WEIGHTS", "");
#else
    ::unsetenv("MAMBA_WEIGHTS");
#endif
    mamba_unlink(good_path);
}

TEST(WeightLoaderPath, RejectsOutOfBoundsDimensions) {
    char tmp_path[512];
    std::filesystem::path p = std::filesystem::temp_directory_path() /
        ("mamba_oob_" + std::to_string(mamba_getpid()) + ".bin");
    std::strncpy(tmp_path, p.string().c_str(), sizeof(tmp_path) - 1);

    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = MAX_D_STATE + 1;  // OVERFLOW
    hdr.d_model = MAX_D_MODEL;
    hdr.d_inner = MAX_D_INNER;
    hdr.d_conv = MAX_D_CONV;
    hdr.num_layers = 1;
    hdr.cond_dim = COND_DIM;
    hdr.weight_data_size = 1024;
    std::FILE* f = std::fopen(tmp_path, "wb");
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::vector<uint8_t> data(1024, 0);
    std::fwrite(data.data(), 1, 1024, f);
    std::fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_path), -14);
    mamba_unlink(tmp_path);
}

TEST(WeightLoaderPath, FitsInStaticWeights) {
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = MAX_D_STATE;
    hdr.d_model = MAX_D_MODEL;
    hdr.d_inner = MAX_D_INNER;
    hdr.d_conv = MAX_D_CONV;
    hdr.num_layers = MAX_LAYERS;
    hdr.cond_dim = COND_DIM;
    EXPECT_TRUE(WeightLoader::fits_in_static_weights_with(hdr));

    hdr.d_state = MAX_D_STATE + 1;
    EXPECT_FALSE(WeightLoader::fits_in_static_weights_with(hdr));
}
