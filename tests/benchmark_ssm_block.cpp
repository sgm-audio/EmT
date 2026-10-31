// tests/benchmark_ssm_block.cpp
//
// Real-time cost benchmark for the SSM inference core.
//
// Design goals (see ARCHITECTURE_FIDELITY.md):
//   1. Report cost in units that map to a product budget: microseconds per
//      sample AND percent of one CPU core at a stated sample rate.
//   2. Gate on an explicit, named budget instead of an unexplained threshold,
//      and return a non-zero exit code when the gate fails so CI can enforce it.
//   3. Measure realistic model shapes, not only the smallest possible one.
//      `--config max` runs the compile-time maxima from src/config.h (a stress
//      test, not a product configuration - see the warnings it prints).
//   4. Be reproducible: median of N repeats, fixed simulated audio duration,
//      machine + build info printed with the numbers.
//
// Examples:
//   ./benchmark_ssm_block                                   # default gate
//   ./benchmark_ssm_block --budget-pct 10 --layers 1         # strategy target
//   ./benchmark_ssm_block --config max --seconds 0.02 --repeats 1
//   ./benchmark_ssm_block --json bench.json --no-gate
//
// Exit code: 0 if all measured configurations meet the budget (or --no-gate),
//            1 if any measured configuration exceeds it.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "core/ssm_block.h"
#include "core/ssm_inference_core.h"
#include "config.h"

#if defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#else
#include <thread>
#endif

using namespace mamba;
using clock_type = std::chrono::steady_clock;

namespace {

// ---------------------------------------------------------------------------
// Model shapes
// ---------------------------------------------------------------------------

struct ModelDims {
    const char* name;
    size_t d_state;
    size_t d_model;
    size_t d_inner;
    size_t nheads;
    size_t headdim;
    size_t mimo_rank;
    size_t num_bc_heads;
    size_t num_rope_angles;
    size_t cond_dim;
};

// Small: what the repo's benchmarks have historically reported. Cheap enough
// for a CI gate.
const ModelDims kSmall = {
    "small", 16, 16, 32, 2, 16, MAX_MIMO_RANK, MAX_NUM_BC_HEADS, 8, COND_DIM
};

// Training-config shapes: these mirror training/configs/*.yaml, i.e. the models
// the exporter actually produces for the plugin. Cost scales ~linearly with
// d_model (the dense in_proj) and ~linearly with nheads*headdim*d_state
// (the recurrence), so these - not the toy "small" shape - are the numbers a
// product decision should be based on.
const ModelDims kTrain4070 = {   // training/configs/laptop_rtx4070.yaml
    "train4070", 16, 1, 64, 2, 32, 1, 1, 8, 32
};
const ModelDims kTrainL4 = {     // training/configs/runpod_l4.yaml
    "trainl4", 32, 1, 256, 4, 64, 1, 1, 16, 128
};

// Max: every compile-time maximum in src/config.h. This is a *stress* shape.
// The dense in_proj is (2*d_inner + 2*nheads*rank*d_state + 3*nheads + angles)
// x d_model, evaluated once per sample, so cost scales roughly linearly in
// d_model and quadratically in the head/state terms. Expect tens of
// microseconds per sample: no plugin should ship at this shape.
const ModelDims kMax = {
    "max", MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_NHEADS, MAX_HEADDIM,
    MAX_MIMO_RANK, MAX_NUM_BC_HEADS, MAX_NUM_ROPE_ANGLES, COND_DIM
};

// ---------------------------------------------------------------------------
// Small argument parser
// ---------------------------------------------------------------------------

struct Options {
    std::vector<const ModelDims*> configs;
    std::vector<size_t> layers = {1, 2};
    std::vector<double> sample_rates = {48000.0};
    std::vector<size_t> block_sizes = {64, 128, 256, 512};
    double budget_pct = 40.0;   // % of one core at the stated sample rate
    int repeats = 5;
    double seconds = 0.25;      // simulated audio duration per measurement
    bool gate = true;
    bool explicit_workload = false;
    std::string json_path;
};

std::vector<std::string> split_csv(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ',') {
            if (!cur.empty()) out.push_back(cur);
            cur.clear();
        } else if (c != ' ') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

bool parse_double_list(const std::string& s, std::vector<double>& out) {
    out.clear();
    for (const auto& tok : split_csv(s)) {
        char* end = nullptr;
        double v = std::strtod(tok.c_str(), &end);
        if (end == tok.c_str() || v <= 0.0) return false;
        out.push_back(v);
    }
    return !out.empty();
}

bool parse_size_list(const std::string& s, std::vector<size_t>& out) {
    out.clear();
    for (const auto& tok : split_csv(s)) {
        char* end = nullptr;
        long long v = std::strtoll(tok.c_str(), &end, 10);
        if (end == tok.c_str() || v <= 0) return false;
        out.push_back(static_cast<size_t>(v));
    }
    return !out.empty();
}

void print_usage() {
    printf(
        "SSM real-time benchmark\n"
        "\n"
        "Usage: benchmark_ssm_block [options]\n"
        "\n"
        "  --config LIST             model shape(s), comma separated (default small):\n"
        "                              small     d_state=16 d_model=16 d_inner=32 (toy)\n"
        "                              train4070 mirrors configs/laptop_rtx4070.yaml\n"
        "                              trainl4   mirrors configs/runpod_l4.yaml\n"
        "                              max       every compile-time MAX_* (stress test)\n"
        "                              all       all of the above\n"
        "  --layers N|a,b|all        layer counts (default 1,2; 'all' = 1..%d)\n"
        "  --sample-rate HZ          shorthand for --sample-rates HZ\n"
        "  --sample-rates a,b        sample rates to report at (default 48000)\n"
        "  --blocks a,b              block sizes to measure (default 64,128,256,512)\n"
        "  --budget-pct PCT          gate: max %% of ONE core at each sample rate\n"
        "                            (default 40; the strategy doc's target is 10)\n"
        "  --repeats N               median of N measurements (default 5)\n"
        "  --seconds SEC             simulated audio per measurement (default 0.25)\n"
        "  --no-gate                 always exit 0 (report-only mode)\n"
        "  --json PATH               also write machine-readable results\n"
        "  --help                    this text\n"
        "\n"
        "Cost model: percent_of_one_core = us_per_sample * sample_rate / 1e4\n"
        "A plugin's real cost is higher than this number: it excludes host buffers,\n"
        "parameter smoothing, GUI, and other plugins in the same session.\n",
        MAX_LAYERS);
}

bool parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto need_value = [&](const char* flag) -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "error: %s requires a value\n", flag);
                return nullptr;
            }
            return argv[++i];
        };

        if (a == "--help" || a == "-h") {
            print_usage();
            std::exit(0);
        } else if (a == "--config") {
            const char* v = need_value("--config");
            if (!v) return false;
            opt.configs.clear();
            for (const std::string& s : split_csv(v)) {
                if (s == "small") opt.configs.push_back(&kSmall);
                else if (s == "train4070") opt.configs.push_back(&kTrain4070);
                else if (s == "trainl4") opt.configs.push_back(&kTrainL4);
                else if (s == "max") opt.configs.push_back(&kMax);
                else if (s == "all") {
                    opt.configs.push_back(&kSmall);
                    opt.configs.push_back(&kTrain4070);
                    opt.configs.push_back(&kTrainL4);
                    opt.configs.push_back(&kMax);
                } else {
                    fprintf(stderr, "error: unknown --config '%s'\n", s.c_str());
                    return false;
                }
            }
            if (opt.configs.empty()) {
                fprintf(stderr, "error: --config needs at least one shape\n");
                return false;
            }
        } else if (a == "--layers") {
            const char* v = need_value("--layers");
            if (!v) return false;
            if (std::string(v) == "all") {
                opt.layers.clear();
                for (size_t n = 1; n <= MAX_LAYERS; ++n) opt.layers.push_back(n);
            } else if (!parse_size_list(v, opt.layers)) {
                fprintf(stderr, "error: bad --layers '%s'\n", v);
                return false;
            }
        } else if (a == "--sample-rate") {
            const char* v = need_value("--sample-rate");
            if (!v || !parse_double_list(v, opt.sample_rates)) {
                fprintf(stderr, "error: bad --sample-rate\n");
                return false;
            }
        } else if (a == "--sample-rates") {
            const char* v = need_value("--sample-rates");
            if (!v || !parse_double_list(v, opt.sample_rates)) {
                fprintf(stderr, "error: bad --sample-rates\n");
                return false;
            }
        } else if (a == "--blocks") {
            const char* v = need_value("--blocks");
            if (!v || !parse_size_list(v, opt.block_sizes)) {
                fprintf(stderr, "error: bad --blocks\n");
                return false;
            }
        } else if (a == "--budget-pct") {
            const char* v = need_value("--budget-pct");
            if (!v) return false;
            opt.budget_pct = std::strtod(v, nullptr);
            if (opt.budget_pct <= 0.0) {
                fprintf(stderr, "error: --budget-pct must be positive\n");
                return false;
            }
        } else if (a == "--repeats") {
            const char* v = need_value("--repeats");
            if (!v) return false;
            opt.repeats = std::atoi(v);
            if (opt.repeats < 1) opt.repeats = 1;
            opt.explicit_workload = true;
        } else if (a == "--seconds") {
            const char* v = need_value("--seconds");
            if (!v) return false;
            opt.seconds = std::strtod(v, nullptr);
            if (opt.seconds <= 0.0) opt.seconds = 0.25;
            opt.explicit_workload = true;
        } else if (a == "--no-gate") {
            opt.gate = false;
        } else if (a == "--json") {
            const char* v = need_value("--json");
            if (!v) return false;
            opt.json_path = v;
        } else {
            fprintf(stderr, "error: unknown option '%s' (try --help)\n", a.c_str());
            return false;
        }
    }
    if (opt.configs.empty()) opt.configs.push_back(&kSmall);
    return true;
}

// ---------------------------------------------------------------------------
// Machine / build info
// ---------------------------------------------------------------------------

std::string cpu_name() {
#if defined(_WIN32)
    char buf[256] = {0};
    DWORD sz = sizeof(buf);
    if (RegGetValueA(HKEY_LOCAL_MACHINE,
                     "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                     "ProcessorNameString", RRF_RT_REG_SZ, nullptr, buf, &sz) == ERROR_SUCCESS) {
        return std::string(buf);
    }
    return "unknown";
#else
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (!f) return "unknown";
    char line[512];
    std::string result = "unknown";
    while (fgets(line, sizeof(line), f)) {
        if (std::strncmp(line, "model name", 10) == 0) {
            const char* colon = std::strchr(line, ':');
            if (colon) {
                std::string v = colon + 1;
                while (!v.empty() && (v.front() == ' ' || v.front() == '\t')) v.erase(v.begin());
                while (!v.empty() && (v.back() == '\n' || v.back() == '\r')) v.pop_back();
                result = v;
            }
            break;
        }
    }
    fclose(f);
    return result;
#endif
}

unsigned hw_threads() {
#if defined(_WIN32)
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return static_cast<unsigned>(si.dwNumberOfProcessors);
#else
    unsigned n = std::thread::hardware_concurrency();
    return n ? n : 1;
#endif
}

void print_environment() {
#if defined(_MSC_VER)
    const char* compiler = "MSVC";
#elif defined(__clang__)
    const char* compiler = "clang " __clang_version__;
#else
    const char* compiler = "gcc " __VERSION__;
#endif

#ifdef NDEBUG
    const char* build = "Release (NDEBUG)";
#else
    const char* build = "Debug/asserts enabled (numbers are NOT representative)";
#endif

    printf("===============================================================\n");
    printf("  SSM real-time benchmark\n");
    printf("===============================================================\n");
    printf("CPU:        %s (%u hardware threads visible)\n", cpu_name().c_str(), hw_threads());
    printf("Compiler:   %s\n", compiler);
    printf("Build:      %s\n", build);
    printf("Compile limits: MAX_D_STATE=%d MAX_D_MODEL=%d MAX_D_INNER=%d "
           "MAX_NHEADS=%d MAX_HEADDIM=%d MAX_LAYERS=%d\n",
           MAX_D_STATE, MAX_D_MODEL, MAX_D_INNER, MAX_NHEADS, MAX_HEADDIM, MAX_LAYERS);
    printf("\n");
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

void fill_input(std::vector<float>& in, double sample_rate) {
    const double w = 2.0 * 3.14159265358979323846 * 440.0 / sample_rate;
    for (size_t i = 0; i < in.size(); ++i)
        in[i] = 0.5f * static_cast<float>(std::sin(w * static_cast<double>(i)));
}

std::unique_ptr<SSMInferenceCore> make_core(const ModelDims& dims, size_t num_layers) {
    MambaConfig cfg{};
    cfg.d_state = dims.d_state;
    cfg.d_model = dims.d_model;
    cfg.d_inner = dims.d_inner;
    cfg.nheads = dims.nheads;
    cfg.headdim = dims.headdim;
    cfg.mimo_rank = dims.mimo_rank;
    cfg.num_bc_heads = dims.num_bc_heads;
    cfg.num_rope_angles = dims.num_rope_angles;
    cfg.num_layers = num_layers;
    cfg.cond_dim = dims.cond_dim;

    auto core = std::make_unique<SSMInferenceCore>();
    core->init(cfg);

    auto w = std::make_unique<SSMWeights>();
    std::memset(w.get(), 0, sizeof(*w));

    // Same deterministic, finite-magnitude weights the previous harness used so
    // timings stay comparable across revisions of this file.
    const size_t in_proj_rows =
        dims.d_inner * 2
        + dims.nheads * dims.mimo_rank * dims.d_state * 2
        + dims.nheads * 3
        + dims.num_rope_angles;
    for (size_t out = 0; out < in_proj_rows && out < SSMWeights::IN_PROJ_OUT; ++out)
        for (size_t j = 0; j < dims.d_model; ++j)
            w->in_proj_w[out][j] = 0.1f;
    for (size_t h = 0; h < dims.nheads; ++h) w->dt_bias[h] = 0.5f;
    for (size_t s = 0; s < dims.d_state; ++s) {
        w->B_norm_weight[s] = 1.0f;
        w->C_norm_weight[s] = 1.0f;
    }
    for (size_t j = 0; j < dims.d_model && j < dims.d_inner; ++j)
        w->out_proj_w[j][j] = 1.0f;

    for (size_t layer = 0; layer < num_layers; ++layer)
        core->set_layer_weights(layer, *w);
    return core;
}

// Returns median microseconds per sample over `repeats` measurements.
double measure_us_per_sample(SSMInferenceCore& core, size_t layers,
                             size_t block_size, double sample_rate,
                             double seconds, int repeats) {
    const size_t num_blocks =
        std::max<size_t>(4, static_cast<size_t>(std::ceil(seconds * sample_rate /
                                                          static_cast<double>(block_size))));

    std::vector<float> in(block_size);
    std::vector<float> out(block_size * MAX_D_MODEL);
    fill_input(in, sample_rate);

    // One untimed warm-up pass, then measure.
    core.process_block(in.data(), out.data(), block_size);
    core.reset();

    std::vector<double> timings;
    timings.reserve(static_cast<size_t>(repeats));
    for (int r = 0; r < repeats; ++r) {
        const auto start = clock_type::now();
        for (size_t b = 0; b < num_blocks; ++b)
            core.process_block(in.data(), out.data(), block_size);
        const auto end = clock_type::now();
        const double total_s = std::chrono::duration<double>(end - start).count();
        timings.push_back(total_s / static_cast<double>(num_blocks * block_size));
        core.reset();
    }
    std::sort(timings.begin(), timings.end());
    (void)layers;
    return timings[timings.size() / 2] * 1e6;
}

struct Result {
    std::string config;
    size_t layers;
    double sample_rate;
    size_t block_size;
    double us_per_sample;
    double core_pct;
    bool pass;
};

void write_json(const std::string& path, const Options& opt,
                const std::vector<Result>& results) {
    std::ofstream f(path);
    if (!f) {
        fprintf(stderr, "warning: could not open %s for writing\n", path.c_str());
        return;
    }
    f << "{\n";
    f << "  \"cpu\": \"" << cpu_name() << "\",\n";
    f << "  \"budget_pct_of_one_core\": " << opt.budget_pct << ",\n";
    f << "  \"repeats\": " << opt.repeats << ",\n";
    f << "  \"seconds_per_measurement\": " << opt.seconds << ",\n";
    f << "  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const Result& r = results[i];
        f << "    {\"config\": \"" << r.config << "\", \"layers\": " << r.layers
          << ", \"sample_rate\": " << r.sample_rate
          << ", \"block_size\": " << r.block_size
          << ", \"us_per_sample\": " << r.us_per_sample
          << ", \"pct_of_one_core\": " << r.core_pct
          << ", \"pass\": " << (r.pass ? "true" : "false") << "}";
        f << (i + 1 == results.size() ? "\n" : ",\n");
    }
    f << "  ]\n}\n";
    printf("Wrote %s\n", path.c_str());
}

} // namespace

int main(int argc, char** argv) {
#if defined(_WIN32)
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#endif
    SSMBlock::apply_ftz_daz();

    Options opt;
    if (!parse_args(argc, argv, opt)) return 2;

    // The max shape is a stress test: with the current per-sample dense
    // in_proj it is orders of magnitude slower than a shippable configuration.
    // Unless the caller asked for a specific workload, shrink the run so the
    // default invocation stays fast.
    bool max_selected = false;
    for (const ModelDims* d : opt.configs) if (d == &kMax) max_selected = true;
    if (max_selected && !opt.explicit_workload) {
        opt.seconds = 0.02;
        opt.repeats = 1;
        printf("[note] '--config max' is a stress shape; using --seconds 0.02 "
               "--repeats 1 (override explicitly to change).\n\n");
    }

    print_environment();

    printf("Budget gate: every measured configuration must stay below %.1f%% of "
           "ONE core at its sample rate.\n", opt.budget_pct);
    printf("             (%% of one core = us_per_sample * sample_rate / 1e4)\n\n");

    std::vector<Result> results;
    bool any_fail = false;

    for (const ModelDims* dims : opt.configs) {
        printf("===============================================================\n");
        printf("Model '%s': d_state=%zu d_model=%zu d_inner=%zu nheads=%zu "
               "headdim=%zu mimo_rank=%zu\n",
               dims->name, dims->d_state, dims->d_model, dims->d_inner,
               dims->nheads, dims->headdim, dims->mimo_rank);
        printf("===============================================================\n");

        for (size_t layers_n : opt.layers) {
            if (layers_n > MAX_LAYERS) {
                printf("[skip] %zu layers exceeds MAX_LAYERS=%d\n", layers_n, MAX_LAYERS);
                continue;
            }
            auto core = make_core(*dims, layers_n);

            for (double sr : opt.sample_rates) {
                const double budget_us = (opt.budget_pct / 100.0) * 1e6 / sr;
                printf("\n--- %zu layer%s @ %.1f kHz  (gate: <= %.2f us/sample = %.1f%% of one core) ---\n",
                       layers_n, layers_n == 1 ? "" : "s", sr / 1000.0, budget_us, opt.budget_pct);
                printf("%-11s %-16s %-12s %-16s %s\n",
                       "Block", "us/sample", "% of one core", "Throughput", "Gate");

                for (size_t bs : opt.block_sizes) {
                    const double us = measure_us_per_sample(*core, layers_n, bs, sr,
                                                            opt.seconds, opt.repeats);
                    const double core_pct = us * sr / 1e4;
                    const bool pass = us <= budget_us;
                    if (!pass) any_fail = true;
                    results.push_back({dims->name, layers_n, sr, bs, us, core_pct, pass});
                    printf("%-11zu %-16.3f %-12.2f%% %-16.0f %s\n",
                           bs, us, core_pct, 1.0 / (us * 1e-6), pass ? "PASS" : "FAIL");
                }
            }
            printf("\n");
        }
    }

    // ---------------------------------------------------------------------
    // Summary
    // ---------------------------------------------------------------------
    size_t n_pass = 0;
    for (const Result& r : results) if (r.pass) ++n_pass;

    printf("===============================================================\n");
    printf("  SUMMARY\n");
    printf("===============================================================\n");
    printf("%-8s %-8s %-10s %-8s %-14s %s\n",
           "Model", "Layers", "Rate", "Block", "us/sample", "Verdict");
    for (const Result& r : results) {
        printf("%-8s %-8zu %-10.1f %-8zu %-14.3f %s\n",
               r.config.c_str(), r.layers, r.sample_rate, r.block_size,
               r.us_per_sample, r.pass ? "PASS" : "FAIL");
    }
    printf("\n%zu / %zu measured configurations meet the %.1f%%-of-one-core budget.\n",
           n_pass, results.size(), opt.budget_pct);
    printf("Headline: worst measured cost is %.3f us/sample at %.1f kHz = %.1f%% of one core.\n",
           [&] {
               double worst = 0.0;
               for (const Result& r : results) worst = std::max(worst, r.us_per_sample);
               return worst;
           }(),
           results.empty() ? 0.0 : results.front().sample_rate / 1000.0,
           [&] {
               double worst = 0.0;
               for (const Result& r : results) worst = std::max(worst, r.core_pct);
               return worst;
           }());
    printf("\nReminder: this covers the SSM core only. A shipped plugin adds host\n"
           "buffer handling, parameter smoothing, GUI work, and neighbours in the\n"
           "session; budget headroom accordingly.\n");

    if (!opt.json_path.empty()) write_json(opt.json_path, opt, results);

    if (any_fail && opt.gate) {
        printf("\nGATE: FAIL (%zu configuration%s over budget)\n",
               results.size() - n_pass, (results.size() - n_pass) == 1 ? "" : "s");
        return 1;
    }
    printf("\nGATE: PASS%s\n", any_fail ? " (failures reported, --no-gate)" : "");
    return 0;
}
