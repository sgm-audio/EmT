#include <gtest/gtest.h>
#include "core/weight_loader.h"
#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include <cstring>
#include <cmath>
#include <cstdio>
#include <unistd.h>

using namespace mamba;

class WeightLoaderTest : public ::testing::Test {
protected:
    char tmp_path_[256];

    void SetUp() override {
        snprintf(tmp_path_, sizeof(tmp_path_), "/tmp/mamba_test_weights_%d.bin", static_cast<int>(getpid()));
    }

    void TearDown() override {
        unlink(tmp_path_);
    }

    size_t floats_per_layer() const {
        uint32_t ds = MAX_D_STATE;
        uint32_t dm = MAX_D_MODEL;
        uint32_t di = MAX_D_INNER;
        uint32_t dc = MAX_D_CONV;
        return ds + ds + ds + ds * di + di * ds + di * ds +
               dc * di + di + di * dm + dm * di + di + dm * ds;
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
    ASSERT_TRUE(create_valid_weight_file(3));

    WeightLoader loader;
    ASSERT_EQ(loader.load(tmp_path_), 0);
    EXPECT_EQ(loader.get_num_layers(), 3u);

    WeightLayerPointers ptrs;
    for (uint32_t i = 0; i < 3; ++i) {
        EXPECT_EQ(loader.get_layer_pointers(i, ptrs), 0);
    }
    EXPECT_NE(loader.get_layer_pointers(3, ptrs), 0);

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
    char tmp_path[256];
    snprintf(tmp_path, sizeof(tmp_path), "/tmp/mamba_integration_%d.bin", static_cast<int>(getpid()));

    size_t fpl = MAX_D_STATE + MAX_D_STATE + MAX_D_STATE +
                 MAX_D_STATE * MAX_D_INNER + MAX_D_INNER * MAX_D_STATE + MAX_D_INNER * MAX_D_STATE +
                 MAX_D_CONV * MAX_D_INNER + MAX_D_INNER +
                 MAX_D_INNER * MAX_D_MODEL + MAX_D_MODEL * MAX_D_INNER + MAX_D_INNER +
                 MAX_D_MODEL * MAX_D_STATE;
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
    unlink(tmp_path);
}
