# EmT Audio-Thread Safety & Static Quality TODO

Audit baseline generated on 2026-11-16. Each item tracks a verified issue in real-time audio-thread safety, numerical stability, AI/DSP parity, CLAP compliance, or static analysis/sanitizers.

## P0 — Critical Audio-Thread & Correctness Bugs

- [x] **P0.1 Eliminate 6.5 MB `process()` `memcpy` + `weight_path` data race**
  - Files: `src/clap/clap_plugin.h`, `src/clap/clap_plugin.cpp`, `src/clap/clap_state.cpp`, `src/clap/preset_codec.cpp`, `src/clap/preset_bank.cpp`
  - Problem: `publish_state_mirror()` copies `8 * sizeof(MambaState)` (~6.5 MB) on every `plugin_process()` call, and reads `m->weight_path` concurrently with `load_weights()` writing it on the main thread.
  - Fix: Remove `weight_path` from `StateMirror` (main thread owns it); only snapshot active/dirty voices or copy `StateMirror` on demand/flag rather than unconditionally copying 6.5 MB every audio block.

- [x] **P0.2 Eliminate 512 KB stack allocation in `SSMInferenceCore::process_block`**
  - Files: `src/core/ssm_inference_core.h`, `src/core/ssm_inference_core.cpp`
  * Problem: `alignas(64) float buf_dm[MAX_BLOCK_SIZE * MAX_D_MODEL]` (512 KB) on the stack inside both `process_block` overloads risks audio-thread stack overflow.
  * Fix: Move `buf_dm_` and `buf_mono_` to preallocated, cache-aligned member scratch buffers in `SSMInferenceCore`.

- [x] **P0.3 Decouple `MAX_MIMO_RANK` file stride from active `mimo_rank_`**
  - Files: `src/core/ssm_block.cpp`, `src/core/weight_loader.cpp`, `src/clap/clap_plugin.cpp`, `src/clap/clap_fx_plugin.cpp`
  * Problem: `load_weights` / `fx_load_weights` never set `core_cfg.mimo_rank = std::max(1u, hdr.mimo_rank)`, leaving `mimo_rank_ = 4` (4× wasted work in `step_rmsnorm_bc` and `step_rope_bc`), while setting `mimo_rank_ = 1` with `dequantize_layer_to` would mis-index `in_proj_w` rows.
  * Fix: Compact `in_proj_w` (or index `B`/`C`/`dd_dt`/`dd_A`/`angle` using `MAX_MIMO_RANK` file stride in `step_in_proj` vs active `mimo_rank_` in `step_rmsnorm_bc`/`step_rope_bc`), and set `core_cfg.mimo_rank` from `hdr.mimo_rank` (defaulting 0 -> 1). Keep Python parity 100% intact.

- [x] **P0.4 Guard `SSMBlock` against `NaN`/`Inf` poisoning**
  - Files: `src/core/ssm_block.cpp`, `src/clap/clap_plugin.cpp`, `src/clap/clap_fx_plugin.cpp`
  * Problem: A single `NaN` or `+Inf` input sample bypasses `softplus()` cutoffs, `A_FLOOR` clamp, `fmod()` in `step_rope_bc`, and `z_val * sigmoid(z_val)` in `step_silu_gate`, permanently poisoning `SSMState`.
  * Fix: Sanitize audio input (`std::isfinite(x) ? std::clamp(x, -1e6f, 1e6f) : 0.0f`), make `softplus()` NaN/Inf-safe, clamp `z_val` in `step_silu_gate`, and replace `std::fmod` in `step_rope_bc` with finite-checked phase wrapping.

## P1 — High-Severity Concurrency, Conditioning, Voice & CLAP Bugs

- [x] **P1.1 Fix `is_processing` data race and `RcuSlot` reclamation / immutability**
  - Files: `src/clap/clap_plugin.h`, `src/clap/clap_plugin.cpp`, `src/clap/clap_fx_plugin.h`, `src/clap/clap_fx_plugin.cpp`, `src/clap/clap_state.cpp`, `src/clap/preset_bank.cpp`, `src/clap/preset_codec.cpp`, `src/common/rcu_slot.h`
  * Problem: `is_processing` is a plain `bool` written on the audio thread and read on the main thread; `RcuSlot::adopt()` overwrites `retired_` if two `publish()` calls occur before `reclaim()`.
  * Fix: Change `is_processing` to `std::atomic<bool>`. Ensure `RcuSlot` never leaks retired pointers on rapid `publish()`/`adopt()`.

- [x] **P1.2 Fix `SSMInferenceCore::set_conditioning` OOB read & `nullptr` handling**
  - Files: `src/core/ssm_inference_core.cpp`
  * Problem: `set_conditioning` always `memcpy`s `COND_DIM` (128) floats instead of `cfg_.cond_dim`, and fails to clear `cond_` or disable conditioning when `cond_vector == nullptr`.
  * Fix: Copy `std::min(cfg_.cond_dim, cond_.size())` floats; when `cond_vector == nullptr`, zero `cond_` and pass `nullptr` to each layer.

- [x] **P1.3 Fix `refresh_conditioning` dropping `f0_probs`/`semantic` and forcing `conditioning_active_` always on**
  - Files: `src/clap/clap_plugin.cpp`
  * Problem: `cf.tempo_bpm = 120.0f` is hardcoded non-zero even when no notes/ARA are active, and `ara_cf.f0_probs` / `ara_cf.semantic` are never copied.
  * Fix: Copy `f0_probs` and `semantic` from `ara_cf`, and only enable non-zero conditioning when a note or ARA frame is active.

- [x] **P1.4 Fix voice stealing click, per-sample envelope rate overwrite, and discarded `d_model` channels**
  - Files: `src/clap/clap_plugin.cpp`
  * Problem: `start_voice()` hard-zeroes `voice.env` on steal (click); `plugin_process()` recomputes `compute_envelope_rate` and overwrites `voice.attack_rate`/`release_rate` every sample; `voice_sum[2..d_model-1]` are computed and discarded.
  * Fix: Smoothly ramp/recompute envelope rates only when parameters change; average/sum `d_model` channels cleanly or document stereo mapping; add anti-click handling or clean voice-steal reset.

- [x] **P1.5 Wire GUI parameter edits to CLAP `out_events` via `param_queue`**
  - Files: `src/clap/clap_plugin.h`, `src/clap/clap_plugin.cpp`, `src/clap/clap_params.cpp`, `src/clap/gui_common.cpp`
  * Problem: `gui_common.cpp::set_param()` mutates `param_values` directly without emitting `CLAP_EVENT_PARAM_VALUE` / gesture events to `out_events`. `param_queue` is unused.
  * Fix: Queue GUI parameter events into `param_queue` and flush them to `out_events` in `params_flush` and `plugin_process`.

- [x] **P1.6 Fix synth `state_load` & `load_preset` hard failure on missing external `weight_path`**
  - Files: `src/clap/clap_state.cpp`, `src/clap/preset_codec.cpp`
  * Problem: Unlike `fx_state_load`, synth `state_load` returns `false` if the saved `weight_path` file is missing, discarding all parameter state.
  * Fix: Probe file existence first; if missing, set `weights_loaded = false` and still restore parameters.

## P2 — Medium-Severity Performance, Numerical & CLAP Polish

- [x] **P2.1 Optimize `step_rope_bc` (hoist per-angle trig out of head loop) and `Seqlocked::read` barriers**
  - Files: `src/core/ssm_block.cpp`, `src/common/seqlock.h`
  * Fix: Compute `angle`/`cos_a`/`sin_a` once per angle `a` (since `angle_ptr[a]` is shared across heads). Use `acquire` loads + `atomic_thread_fence(acquire)` in `Seqlocked::read`.

- [x] **P2.2 Guard `SSMBlock::process_block` with `ScopedFtzDaz`**
  - Files: `src/core/ssm_block.cpp`

- [x] **P2.3 Fix `fp16_to_fp32` subnormal flushing**
  - Files: `src/core/weight_loader.h`
  * Fix: Properly normalize FP16 subnormals (`exponent == 0 && mantissa != 0`) instead of flushing to `0.0f`.

- [x] **P2.4 Fix `fx_init` throwing `new`, `fx_process` per-sample atomics, and `tail_get` release tail**
  - Files: `src/clap/clap_fx_plugin.cpp`, `src/clap/clap_tail.cpp`
  * Fix: Use `new (std::nothrow)` + null check in `fx_init`; hoist atomic loads out of `fx_process` sample loop; base `tail_get` on `PARAM_RELEASE`.

- [x] **P2.5 Assert `std::atomic<std::thread::id>::is_always_lock_free`**
  - Files: `src/clap/clap_thread_check.h`

## P3 — Static Analysis, Warnings & Sanitizers

- [x] **P3.1 Add `.clang-tidy` and stricter compiler warnings (`-Wshadow`, `-Wdouble-promotion`, `-Wfloat-equal`)**
  - Files: `.clang-tidy`, `CMakeLists.txt`
- [x] **P3.2 Add ASan/UBSan and TSan presets/scripts and verify all tests pass under sanitizers**
  - Files: `scripts/ci.sh` or `CMakeLists.txt` / `.github/workflows/ci.yml`
