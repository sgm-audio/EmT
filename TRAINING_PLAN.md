# Mamba-3 Plugin — Training Plans

This document describes how to train Mamba-3 weights for the CLAP plugin on
two targets: a local **RTX 4070 laptop** and a cloud **Runpod L4 (24 GB)**.
The training code lives in `training/` and re-uses the same reference model
that `scripts/export_mamba_weights.py` and the C++ `WeightLoader` consume.

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
- **Objective**: next-sample prediction on raw mono waveforms.
  Input `x[:, :-1]` predicts target `y[:, 1:]`; loss is MSE.
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

## Notes and limitations

- `d_model=1` is tuned for raw audio.  For latent tokens, update `d_model` and
  make sure your dataset returns `(seq_len, d_model)` frames.
- The C++ plugin currently uses only rank-0 B/C projection and ignores the
  unused `dt_film_proj_w` weights.  Training still exports them for forward
  compatibility.
- `process_block()` in C++ is still a per-sample loop stub; real-time block
  processing is not yet optimized.
- If you change `MAX_*` in `config.h`, rebuild all C++ targets; the
  `WeightLoader` rejects files whose header dims exceed the compiled limits.

## Verification checklist

- [ ] `python training/train.py --smoke-test --steps 5`
- [ ] Exported `.bin` passes `python scripts/validate_export.py <path>`
- [ ] C++ tests pass after copying `config.h` and rebuilding
- [ ] `benchmark_ssm_block` shows single-layer time well under 10 µs/sample
