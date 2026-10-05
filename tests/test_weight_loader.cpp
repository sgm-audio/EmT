#include <gtest/gtest.h>
#include "core/weight_loader.h"
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "fixtures/weight_fixture.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <limits>

using namespace mamba;

class WeightLoaderTest : public ::testing::Test {
protected:
    // RAII temp path (unlinked on destruction); writers below share
    // tests/fixtures/weight_fixture.h with the other suites.
    test::TempWeightFile tmp_file_{"mamba3_test_weights_"};

    // `mimo_rank` defaults to 0 = legacy file written before the field existed
    // (WeightLoader treats 0 as rank 1).  Pass > 1 to exercise the D3 refusal.
    bool create_valid_weight_file(uint32_t num_layers = 1, uint32_t mimo_rank = 0) {
        test::WeightFileSpec spec;
        spec.num_layers = num_layers;
        spec.mimo_rank = mimo_rank;
        return test::write_weight_file(tmp_file_.path, spec);
    }
};

TEST_F(WeightLoaderTest, LoadValidFile) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    LoadError rc = loader.load(tmp_file_.path);
    EXPECT_EQ(rc, LoadError::Ok);
    EXPECT_TRUE(loader.is_loaded());
    EXPECT_EQ(loader.get_num_layers(), 1u);

    loader.unload();
    EXPECT_FALSE(loader.is_loaded());
}

// Deviation D3: rank-R MIMO is reserved in the format but the forward pass
// implements rank-1 SISO per head.  A rank-R file must be refused loudly, not
// silently run as rank 1.
TEST_F(WeightLoaderTest, RejectsUnimplementedMimoRank) {
    ASSERT_TRUE(create_valid_weight_file(/*num_layers=*/1, /*mimo_rank=*/2));

    WeightLoader loader;
    LoadError rc = loader.load(tmp_file_.path);
    EXPECT_EQ(rc, LoadError::MimoRankRefused) << "rank-2 MIMO file must be refused with the dedicated "
                          "error code, not loaded";
    EXPECT_FALSE(loader.is_loaded());
    EXPECT_FALSE(loader.fits_in_static_weights());
}

// Files written before the mimo_rank field existed have a zero `reserved` slot
// there and must keep loading as rank 1.
TEST_F(WeightLoaderTest, LegacyHeaderWithoutMimoFieldIsRankOne) {
    ASSERT_TRUE(create_valid_weight_file(/*num_layers=*/1, /*mimo_rank=*/0));

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_file_.path), LoadError::Ok);
    EXPECT_TRUE(loader.is_loaded());
    EXPECT_EQ(loader.header().mimo_rank, 0u);
    EXPECT_TRUE(loader.fits_in_static_weights());
}

TEST_F(WeightLoaderTest, LoadNonexistentFile) {
    WeightLoader loader;
    LoadError rc = loader.load("/tmp/mamba3_nonexistent_weights.bin");
    EXPECT_NE(rc, LoadError::Ok);
    EXPECT_FALSE(loader.is_loaded());
}

TEST_F(WeightLoaderTest, ValidateDimensions) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_file_.path), LoadError::Ok);

    const auto& hdr = loader.header();
    EXPECT_TRUE(loader.validate_dimensions(
        hdr.d_state, hdr.d_model, hdr.d_inner,
        hdr.nheads, hdr.headdim, hdr.num_layers));
    EXPECT_FALSE(loader.validate_dimensions(
        hdr.d_state + 1, hdr.d_model, hdr.d_inner,
        hdr.nheads, hdr.headdim, hdr.num_layers));

    loader.unload();
}

TEST_F(WeightLoaderTest, GetLayerPointers) {
    ASSERT_TRUE(create_valid_weight_file());

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_file_.path), LoadError::Ok);

    WeightLayerPointers ptrs;
    LoadError rc = loader.get_layer_pointers(0, ptrs);
    EXPECT_EQ(rc, LoadError::Ok);
    EXPECT_NE(ptrs.in_proj_w, nullptr);
    EXPECT_NE(ptrs.dt_bias, nullptr);
    EXPECT_NE(ptrs.B_bias, nullptr);
    EXPECT_NE(ptrs.C_bias, nullptr);
    EXPECT_NE(ptrs.B_norm_weight, nullptr);
    EXPECT_NE(ptrs.C_norm_weight, nullptr);
    EXPECT_NE(ptrs.mimo_x, nullptr);
    EXPECT_NE(ptrs.mimo_z, nullptr);
    EXPECT_NE(ptrs.mimo_o, nullptr);
    EXPECT_NE(ptrs.D_proj, nullptr);
    EXPECT_NE(ptrs.out_proj_w, nullptr);
    EXPECT_NE(ptrs.film_proj_w, nullptr);
    EXPECT_NE(ptrs.dt_film_proj_w, nullptr);

    EXPECT_FLOAT_EQ(ptrs.in_proj_w[0], -1.0f);

    rc = loader.get_layer_pointers(1, ptrs);
    EXPECT_NE(rc, LoadError::Ok);

    loader.unload();
}

TEST_F(WeightLoaderTest, LoadMultiLayer) {
    const uint32_t kNumLayers = 2;

    ASSERT_TRUE(create_valid_weight_file(kNumLayers));

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_file_.path), LoadError::Ok);
    EXPECT_EQ(loader.get_num_layers(), kNumLayers);

    WeightLayerPointers ptrs;
    for (uint32_t i = 0; i < kNumLayers; ++i) {
        EXPECT_EQ(loader.get_layer_pointers(i, ptrs), LoadError::Ok);
    }
    EXPECT_NE(loader.get_layer_pointers(kNumLayers, ptrs), LoadError::Ok);

    loader.unload();
}

TEST_F(WeightLoaderTest, BadMagic) {
    FILE* f = fopen(tmp_file_.path, "wb");
    ASSERT_NE(f, nullptr);
    WeightFileHeader hdr{};
    hdr.magic = 0xDEADBEEF;
    hdr.version = WEIGHT_VERSION;
    hdr.num_layers = 1;
    hdr.weight_data_size = 0;
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_file_.path), LoadError::BadMagic);
}

TEST_F(WeightLoaderTest, BadVersion) {
    FILE* f = fopen(tmp_file_.path, "wb");
    ASSERT_NE(f, nullptr);
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = 999;
    hdr.num_layers = 1;
    hdr.weight_data_size = 0;
    fwrite(&hdr, sizeof(hdr), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_file_.path), LoadError::BadVersion);
}

TEST_F(WeightLoaderTest, FileTooSmall) {
    FILE* f = fopen(tmp_file_.path, "wb");
    ASSERT_NE(f, nullptr);
    uint32_t bad = 0;
    fwrite(&bad, sizeof(bad), 1, f);
    fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_file_.path), LoadError::FileTooSmall);
}

TEST(WeightIntegration, LoadWeightsThenProcessNonZero) {
    test::TempWeightFile tmp("mamba3_integration_");
    const char* tmp_path = tmp.path;

    uint32_t ds = MAX_D_STATE, dm = MAX_D_MODEL, di = MAX_D_INNER;
    uint32_t nh = MAX_NHEADS, mr = MAX_MIMO_RANK;
    uint32_t hd = MAX_HEADDIM;
    size_t in_proj_out = test::in_proj_out_for(di, nh, ds);

    test::WeightFileSpec spec;
    spec.num_layers = 2;
    spec.mark_first = false;  // uniform 0.1f payload, as before
    ASSERT_TRUE(test::write_weight_file(tmp_path, spec));

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path), LoadError::Ok);

    const auto& lhdr = loader.header();
    ASSERT_TRUE(loader.validate_dimensions(
        lhdr.d_state, lhdr.d_model, lhdr.d_inner,
        lhdr.nheads, lhdr.headdim, lhdr.num_layers));

    MambaConfig cfg = MambaConfig::max_config();
    cfg.num_layers = 2;
    auto core_ptr = std::make_unique<SSMInferenceCore>();
    core_ptr->init(cfg);
    auto& core = *core_ptr;

    for (uint32_t layer = 0; layer < 2; ++layer) {
        WeightLayerPointers ptrs;
        ASSERT_EQ(loader.get_layer_pointers(layer, ptrs), LoadError::Ok);

        SSMWeights w{};
        std::memcpy(w.in_proj_w, ptrs.in_proj_w, in_proj_out * dm * sizeof(float));
        std::memcpy(w.dt_bias, ptrs.dt_bias, nh * sizeof(float));
        std::memcpy(w.B_bias, ptrs.B_bias, nh * mr * ds * sizeof(float));
        std::memcpy(w.C_bias, ptrs.C_bias, nh * mr * ds * sizeof(float));
        std::memcpy(w.B_norm_weight, ptrs.B_norm_weight, ds * sizeof(float));
        std::memcpy(w.C_norm_weight, ptrs.C_norm_weight, ds * sizeof(float));
        std::memcpy(w.mimo_x, ptrs.mimo_x, nh * mr * hd * sizeof(float));
        std::memcpy(w.mimo_z, ptrs.mimo_z, nh * mr * hd * sizeof(float));
        std::memcpy(w.mimo_o, ptrs.mimo_o, nh * mr * hd * sizeof(float));
        std::memcpy(w.D_proj, ptrs.D_proj, nh * sizeof(float));
        std::memcpy(w.out_proj_w, ptrs.out_proj_w, dm * di * sizeof(float));

        core.set_layer_weights(layer, w);
    }

    alignas(64) float outbuf[MAX_D_MODEL];
    core.process_sample(1.0f, outbuf);
    EXPECT_FALSE(std::isnan(outbuf[0]));
    EXPECT_FALSE(std::isinf(outbuf[0]));

    loader.unload();
}

// FP16 tests unchanged
TEST(FP16, ZeroAndSignedZero) {
    EXPECT_EQ(fp16_to_fp32(0x0000), 0.0f);
    EXPECT_EQ(fp16_to_fp32(0x8000), -0.0f);
}

TEST(FP16, OneAndMinusOne) {
    EXPECT_EQ(fp16_to_fp32(0x3C00), 1.0f);
    EXPECT_EQ(fp16_to_fp32(0xBC00), -1.0f);
}

TEST(FP16, TwoAndHalf) {
    EXPECT_EQ(fp16_to_fp32(0x4000), 2.0f);
    EXPECT_EQ(fp16_to_fp32(0x3800), 0.5f);
}

TEST(FP16, InfAndNaN) {
    EXPECT_EQ(fp16_to_fp32(0x7C00), std::numeric_limits<float>::infinity());
    EXPECT_EQ(fp16_to_fp32(0xFC00), -std::numeric_limits<float>::infinity());
    float nan_val = fp16_to_fp32(0x7E00);
    EXPECT_TRUE(std::isnan(nan_val));
}

TEST(FP16, RoundTripKnownValues) {
    for (uint32_t bits = 0; bits < 0x10000; bits += 0x101) {
        uint16_t h = static_cast<uint16_t>(bits);
        float f = fp16_to_fp32(h);
        if (std::isnan(f) || std::isinf(f)) continue;
        uint32_t exp_field = (bits >> 10) & 0x1F;
        if (exp_field == 0) continue;
        uint32_t mant = bits & 0x3FF;
        uint32_t sign = (bits >> 15) & 0x1;
        uint32_t fp32_bits = (sign << 31) | (((exp_field - 15 + 127) & 0xFF) << 23) | (mant << 13);
        float expected;
        std::memcpy(&expected, &fp32_bits, sizeof(expected));
        EXPECT_EQ(f, expected) << "Mismatch at bits=0x" << std::hex << bits;
    }
}

namespace {
// Movable RAII temp file; callers keep using `qf.path`.
using QFile = test::TempWeightFile;

QFile make_i8_per_layer_file(float scale, int32_t zp) {
    QFile tmp("mamba3_q8_");

    uint32_t ds = MAX_D_STATE, dm = MAX_D_MODEL, di = MAX_D_INNER;
    uint32_t nh = MAX_NHEADS;
    uint32_t hd = MAX_HEADDIM, cd = COND_DIM;
    size_t bytes_per_layer = test::floats_per_layer_for(ds, dm, di, nh, hd, cd);
    size_t q_meta_size = sizeof(float) + sizeof(int32_t) + 8;
    size_t weight_data_size = bytes_per_layer;

    std::FILE* f = std::fopen(tmp.path, "wb");
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = ds; hdr.d_model = dm; hdr.d_inner = di;
    hdr.nheads = nh; hdr.num_layers = 1; hdr.headdim = hd;
    hdr.cond_dim = cd;
    hdr.quantization = static_cast<uint32_t>(QuantFormat::Int8PerLayer);
    hdr.q_metadata_size = static_cast<uint32_t>(q_meta_size);
    hdr.weight_data_size = static_cast<uint64_t>(weight_data_size);
    std::fwrite(&hdr, sizeof(hdr), 1, f);

    float s = scale;
    int32_t z = zp;
    uint8_t pad[8] = {0};
    std::fwrite(&s, sizeof(s), 1, f);
    std::fwrite(&z, sizeof(z), 1, f);
    std::fwrite(pad, 1, 8, f);

    std::vector<int8_t> data(weight_data_size);
    for (size_t i = 0; i < data.size(); ++i) {
        int32_t v = static_cast<int32_t>(i % 5) + zp;
        if (v < -128) v = -128;
        if (v > 127) v = 127;
        data[i] = static_cast<int8_t>(v);
    }
    std::fwrite(data.data(), 1, weight_data_size, f);
    std::fclose(f);

    return tmp;
}
} // namespace

TEST(WeightQuantization, Int8PerLayerRoundTrip) {
    QFile qf = make_i8_per_layer_file(0.01f, 0);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), LoadError::Ok);
    ASSERT_EQ(loader.quant_format(), QuantFormat::Int8PerLayer);

    LayerQParam qp;
    ASSERT_EQ(loader.get_layer_qparam(0, qp), LoadError::Ok);
    EXPECT_FLOAT_EQ(qp.scale, 0.01f);
    EXPECT_EQ(qp.zero_point, 0);

    SSMWeights w{};
    ASSERT_EQ(loader.dequantize_layer_to(0, w), LoadError::Ok);

    EXPECT_NEAR(w.in_proj_w[0][0], 0.0f, 1e-6f);  // data[0]=0, out = 0.01*(0-0)=0
    EXPECT_NEAR(w.in_proj_w[0][1], 0.01f, 1e-6f); // data[1]=1, out = 0.01*(1-0)=0.01

    loader.unload();
}

TEST(WeightQuantization, Int8PerLayerOffsetAndScale) {
    QFile qf = make_i8_per_layer_file(0.5f, 10);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), LoadError::Ok);

    SSMWeights w{};
    ASSERT_EQ(loader.dequantize_layer_to(0, w), LoadError::Ok);
    EXPECT_NEAR(w.in_proj_w[0][0], 0.0f, 1e-6f);  // data[0]=10, out=0.5*(10-10)=0
    EXPECT_NEAR(w.in_proj_w[0][1], 0.5f, 1e-6f);  // data[1]=11, out=0.5*(11-10)=0.5
    loader.unload();
}

TEST(WeightQuantization, RejectsWrongFormatAccess) {
    QFile qf = make_i8_per_layer_file(0.01f, 0);
    WeightLoader loader;
    ASSERT_EQ(loader.load(qf.path), LoadError::Ok);

    WeightLayerPointers fp;
    EXPECT_NE(loader.get_layer_pointers(0, fp), LoadError::Ok);
    WeightLayerPointersF16 f16p;
    EXPECT_NE(loader.get_layer_pointers_f16(0, f16p), LoadError::Ok);
    TensorQParam tqp;
    EXPECT_NE(loader.get_tensor_qparam(tqp), LoadError::Ok);

    loader.unload();
}

// Path resolution tests
TEST(WeightLoaderPath, MAMBAWeightsEnvVar) {
    test::TempWeightFile tmp("mamba3_env_");
    const char* tmp_path = tmp.path;

    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = 1; hdr.d_model = 1; hdr.d_inner = 1;
    hdr.nheads = 1; hdr.num_layers = 1; hdr.headdim = 1;
    hdr.cond_dim = 1;
    // Must match bytes_per_layer() which uses MAX_MIMO_RANK and MAX_NUM_ROPE_ANGLES
    uint32_t ds = hdr.d_state, dm = hdr.d_model, di = hdr.d_inner;
    uint32_t nh = hdr.nheads, hd = hdr.headdim, cd = hdr.cond_dim;
    size_t fpl = test::floats_per_layer_for(ds, dm, di, nh, hd, cd);
    hdr.weight_data_size = fpl * sizeof(float);
    std::FILE* f = std::fopen(tmp_path, "wb");
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::vector<float> data(fpl, 0.5f);
    std::fwrite(data.data(), sizeof(float), fpl, f);
    std::fclose(f);

#ifdef _WIN32
    _putenv_s("EMT_WEIGHTS", tmp_path);
#else
    ::setenv("EMT_WEIGHTS", tmp_path, 1);
#endif
    WeightLoader loader;
    LoadError rc = loader.load();
    EXPECT_EQ(rc, LoadError::Ok);
    EXPECT_STREQ(loader.source_path(), tmp_path);
    loader.unload();
#ifdef _WIN32
    _putenv_s("EMT_WEIGHTS", "");
#else
    ::unsetenv("EMT_WEIGHTS");
#endif
}

TEST(WeightLoaderPath, RejectsOutOfBoundsDimensions) {
    test::TempWeightFile tmp("mamba3_oob_");
    const char* tmp_path = tmp.path;

    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = MAX_D_STATE + 1;  // overflow
    hdr.d_model = MAX_D_MODEL;
    hdr.d_inner = MAX_D_INNER;
    hdr.nheads = MAX_NHEADS;
    hdr.num_layers = 1;
    hdr.headdim = MAX_HEADDIM;
    hdr.cond_dim = COND_DIM;
    hdr.weight_data_size = 1024;
    std::FILE* f = std::fopen(tmp_path, "wb");
    std::fwrite(&hdr, sizeof(hdr), 1, f);
    std::vector<uint8_t> data(1024, 0);
    std::fwrite(data.data(), 1, 1024, f);
    std::fclose(f);

    WeightLoader loader;
    EXPECT_EQ(loader.load(tmp_path), LoadError::DimsExceedLimits);
}

TEST(WeightLoaderPath, FitsInStaticWeights) {
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = MAX_D_STATE; hdr.d_model = MAX_D_MODEL; hdr.d_inner = MAX_D_INNER;
    hdr.nheads = MAX_NHEADS; hdr.num_layers = MAX_LAYERS; hdr.headdim = MAX_HEADDIM;
    hdr.cond_dim = COND_DIM;
    EXPECT_TRUE(WeightLoader::fits_in_static_weights_with(hdr));

    hdr.d_state = MAX_D_STATE + 1;
    EXPECT_FALSE(WeightLoader::fits_in_static_weights_with(hdr));
}

// Regression pin for the model-load path. `fits_in_static_weights_with` asks
// "is this file within the compiled limits?" - which is what the plugins need.
// `validate_dimensions` asks "does this file's shape equal the shape I pass?",
// so calling it with the constructor's placeholder num_layers (1) rejected every
// multi-layer model, including the ones training/configs/*.yaml produce. Both
// plugins now call the predicate below; this test documents the difference.
TEST(WeightLoaderPath, MultiLayerModelIsLoadable) {
    WeightFileHeader hdr{};
    hdr.magic = WEIGHT_MAGIC;
    hdr.version = WEIGHT_VERSION;
    hdr.d_state = 16; hdr.d_model = 1; hdr.d_inner = 64;
    hdr.nheads = 2; hdr.num_layers = 3; hdr.headdim = 32;
    hdr.cond_dim = 32;

    EXPECT_TRUE(WeightLoader::fits_in_static_weights_with(hdr))
        << "a 3-layer model within the compiled limits must be accepted";

    hdr.num_layers = MAX_LAYERS + 1;
    EXPECT_FALSE(WeightLoader::fits_in_static_weights_with(hdr))
        << "a model with more layers than MAX_LAYERS must be rejected";
}
