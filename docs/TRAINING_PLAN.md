# SSM Plugin — Training Plans

This document describes how to train weights for the CLAP plugin on two targets:
a local **RTX 4070 laptop** and a cloud **Runpod L4 (24 GB)**. The training code
lives in `training/` and re-uses the same reference model that
`scripts/export_mamba_weights.py` and the C++ `WeightLoader` consume.

> The architecture is **Mamba-3-inspired**, not a reproduction of the paper: the
> recurrence is a two-term bilinear update with a real state, and the MIMO path is
> inert. See [`ARCHITECTURE_FIDELITY.md`](ARCHITECTURE_FIDELITY.md). Train what
> the plugin runs - the parity test in `tests/test_cpp_parity.cpp` exists to keep
> the Python reference and the C++ inference path honest with each other.

## Quick start

```bash
# Smoke test (synthetic data, no dataset required)
python training/train.py --smoke-test --steps 5

# Real training on a folder of audio files
python training/train.py \
    --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/wav \
    --output-dir runs/run1

# Override the knobs you will tweak most
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/wav --output-dir runs/run2 \
    --batch-size 2 --grad-accum 4 --num-workers 2 --no-amp

# Validate the exported .bin round-trips through PyTorch
python training/train.py --config training/configs/laptop_rtx4070.yaml \
    --dataset-dir data/wav --output-dir runs/run3 --validate
```

## What the trainer does

- **Model**: `training/train.py` imports the reference Mamba-3 model from
  `scripts/export_mamba_weights.py` (single source of truth, no drift).
- **Objective**: selectable with `--loss` (or the `loss:` key in the config):
  - `hybrid` (default in the shipped configs) = pre-emphasis L1 + SI-SDR +
    multi-resolution STFT. This is the task-matched objective for effects and
    restoration; see `training/losses.py`.
  - `mse` = the legacy next-sample MSE, kept for reproducibility.
  - `l1`, `sisdr`, `mrstft` = single-term variants.
  Weights are configurable: `loss_weights: {time, sisdr, stft}`,
  `loss_preemph`, `loss_fft_sizes`/`loss_hop_sizes`/`loss_win_lengths`.
  Run `python3 training/losses.py` for a self-test of the objective's properties.
- **Two training modes**:
  - *self-supervised* (default): next-sample prediction on raw mono waveforms
    (`x[:, :-1]` predicts `y[:, 1:]`); use it for a generative/synthesis model.
  - *paired*: if `--dataset-dir` contains `dry/` and `wet/` subfolders (or
    `--paired` is passed), the model learns the mapping dry -> wet, which is what
    an FX plugin actually has to do. Sample *i* of dry must align with sample *i*
    of wet: the model is causal with zero look-ahead, so any offset in the wet
    render is learned as part of the effect.
- **Decoding**: `soundfile`, then `torchaudio`, then the stdlib `wave` module
  (uncompressed PCM WAV). Recent torchaudio builds need TorchCodec, so the
  stdlib fallback keeps a bare machine working.
- **Fail-safe**: if no optimisation step runs (all files skipped, or windows
  shorter than `seq_len`), the trainer raises instead of exporting untrained
  weights.
- **Conditioning**: a global `(batch, cond_dim)` FiLM vector broadcast across
  time.  Currently zero-filled; extend `AudioWaveDataset` to inject pitch,
  velocity, instrument one-hot, etc.
- **Export**: after training, the script writes the flat-binary `.bin` that
  `src/core/weight_loader.cpp` reads **and** regenerates `src/config.h` with
  matching `MAX_*` constants.
- **Validation** (`--validate`): runs `scripts/validate_export.py` to confirm
  the binary round-trips through PyTorch without NaN/Inf.

## Hardware targets

| Target | VRAM | Config | Effective batch | Seq len | Params (approx) |
|--------|------|--------|-----------------|---------|-----------------|
| RTX 4070 Laptop | 8 GB | `configs/laptop_rtx4070.yaml` | 8 | 2048 | ~160 K |
| Runpod L4 | 24 GB | `configs/runpod_l4.yaml` | 16 | 4096 | ~3.4 M |

Both configs leave headroom for mixed precision (`amp: true`).  If you hit OOM,
reduce `batch_size` or `seq_len` first; `grad_accum` can compensate for batch
size.

> **⚠️ Measured real-time reality (2026-10-04).** Training a model the plugin
> cannot run is the most expensive mistake available here. Measured with
> `tests/benchmark_ssm_block.cpp` on a 2-core Xeon container (core-only cost,
> percent of ONE core at 48 kHz):
>
> | Config | 1 layer | 3 layers | 6 layers |
> |---|---|---|---|
> | `laptop_rtx4070` (d_inner 64, 2 heads) | 4.3 us = 20 % | 8.4-9.6 us = 40-46 % | 15.5-16.7 us = 74-80 % |
> | `runpod_l4` (d_inner 256, 4 heads) | 18.8-20.1 us = **90-96 %** | 54-73 us = 260-350 % | 114-117 us = 546-564 % |
>
> The L4 config is **not real-time at any depth** with the current scalar
> recurrence: one layer alone consumes essentially the whole audio thread. Shrink
> `d_inner`/`nheads`/`headdim` (or wait for the SIMD block path) before training
> anything you intend to ship, and re-measure with:
>
> ```bash
> ./build/tests/benchmark_ssm_block --config train4070,trainl4 --layers 1,3,6 --json bench.json
> ```
>
> The exporter writes whatever shape you trained, and the plugin re-initialises
> itself to that shape at load time - so a heavy config ships as a heavy plugin.

## RTX 4070 Laptop plan

1. **Install deps**
   ```bash
   uv pip install torch torchaudio numpy pyyaml tqdm
   ```
   (Windows native is fine; no separate `mamba-ssm` CUDA kernels are required.)

2. **Prepare data**
   ```bash
   mkdir -p data/wav
   # Copy mono/stereo WAV/FLAC/MP3 files here.
   ```

3. **Train**
   ```bash
   python training/train.py \
       --config training/configs/laptop_rtx4070.yaml \
       --dataset-dir data/wav \
       --output-dir runs/rtx4070
   ```

4. **Export / test in plugin**
   The run already writes `runs/rtx4070/plugin_weights.bin` and
   `runs/rtx4070/config.h`.  Copy them into place:
   ```bash
   cp runs/rtx4070/config.h src/config.h
   cp runs/rtx4070/plugin_weights.bin models/test_model.bin
   cmake --build build --config Release
   ./build/tests/Release/test_cpp_parity    # or test_ssm_block
   ```

## Runpod L4 cloud plan

1. **Spin up a PyTorch L4 pod** (24 GB VRAM, CUDA capable).

2. **Install deps**
   ```bash
   uv pip install torch torchaudio numpy pyyaml tqdm
   ```

3. **Upload data** and start training:
   ```bash
   python training/train.py \
       --config training/configs/runpod_l4.yaml \
       --dataset-dir data/wav \
       --output-dir runs/runpod_l4
   ```

4. **Download artifacts**
   - `runs/runpod_l4/final.pt` — full PyTorch checkpoint
   - `runs/runpod_l4/plugin_weights.bin` — plugin weight file
   - `runs/runpod_l4/config.h` — matching C++ compile-time limits

## Files changed / added

- `training/train.py` — single, debuggable trainer + smoke-test mode.
- `training/configs/laptop_rtx4070.yaml` — local 8 GB config.
- `training/configs/runpod_l4.yaml` — cloud 24 GB config.
- `training/pyproject.toml` — dependencies for `uv`.
- `scripts/export_mamba_weights.py` — made RoPE in-place ops training-safe.
- `TRAINING_PLAN.md` — this document.

## Known limitations

- **State renormalization — removed (D5, closed 2026-10-04)**: there is no
  periodic state scaling any more, in training or in the plugin. Training windows
  therefore no longer need to be longer than 4096 samples for that reason; the
  stability argument is in `ARCHITECTURE_FIDELITY.md` D5 and is guarded by
  `tests/test_stability.cpp`.
- **MIMO**: `mimo_rank > 1` is refused up front — by `training/train.py` before
  training and by the exporter before writing a file (deviation D3, closed
  2026-10-04). Keep `mimo_rank: 1` in your config until the rank-R reduction is
  implemented (F3 in `ARCHITECTURE_FIDELITY.md` §5).
- **Δ-FiLM is live (D6, closed 2026-10-04)**: `dt_film_proj` now receives
  gradients and modulates Δ at inference with the same convention as the input
  FiLM. Do not disable it in the config: training and inference must agree.
- **Hyper-parameter sweep order**: the real-time budget is set by the *shape*
  (`d_inner`, `nheads`, `headdim`, `d_state`, depth). Fix the shape from a
  benchmark first, then spend GPU hours on data and loss weights.

## Notes and limitations

- `d_model=1` is tuned for raw audio.  For latent tokens, update `d_model` and
  make sure your dataset returns `(seq_len, d_model)` frames.
- The C++ plugin uses only rank-0 B/C projection (rank > 1 is refused, D3) and
  since the D6 closure it *does* use `dt_film_proj_w` to modulate Δ.  Any change
  to the conditioning path must land in the trainer and the runtime together.
- `process_block()` in C++ is still a per-sample loop stub; real-time block
  processing is not yet optimized.
- If you change `MAX_*` in `config.h`, rebuild all C++ targets; the
  `WeightLoader` rejects files whose header dims exceed the compiled limits.

## Verification checklist

- [ ] `python training/train.py --smoke-test --steps 5`
- [ ] Exported `.bin` passes `python scripts/validate_export.py <path>`
- [ ] C++ tests pass after copying `config.h` and rebuilding
- [ ] `benchmark_ssm_block --config train4070 --layers <your depth>` reports a
  measured per-sample cost inside the budget you are willing to ship
  (`--budget-pct`), on the target machine, quoted from its JSON output
- [ ] `python training/losses.py` self-test passes
