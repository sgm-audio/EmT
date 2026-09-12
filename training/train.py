#!/usr/bin/env python3
"""
training/train.py

Single-file, debuggable trainer for the Mamba-3 CLAP plugin model.

What it does
------------
1. Builds the same Mamba-3 architecture the C++ WeightLoader consumes
   (imported from scripts/export_mamba_weights.py, so there is no drift).
2. Trains on a folder of audio files (or synthetic sinusoids when none given).
3. Periodically checkpoints and, on finish, exports:
     - <output-dir>/plugin_weights.bin   (flat-binary for the C++ plugin)
     - <output-dir>/config.h             (matching MAX_* constants)

Usage
-----
    # Real training
    python training/train.py \
        --config training/configs/laptop_rtx4070.yaml \
        --dataset-dir data/wav --output-dir runs/run1

    # Quick 5-step smoke test (no dataset needed)
    python training/train.py --smoke-test --steps 5

    # Override anything from the CLI
    python training/train.py --config training/configs/laptop_rtx4070.yaml \
        --batch-size 2 --grad-accum 4 --num-workers 2 --no-amp \
        --dataset-dir data/wav --output-dir runs/run2

CLI overrides (these are the ones you will tweak the most):
    --batch-size     samples per forward/backward step
    --grad-accum     gradient accumulation steps (effective batch = B * G)
    --num-workers    0 = in-process (easiest to debug); >0 = DataLoader workers
    --no-amp         disable automatic mixed precision (helps debug NaN/Inf)
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
import torch.nn.functional as F
import yaml

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_PROJECT_ROOT, "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from export_mamba_weights import (  # noqa: E402
    Mamba3Model,
    export_model,
    generate_config_h,
)

AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".aiff"}


# ---------------------------------------------------------------------------
# Model wrapper (adds conditioning-aware forward to the reference model)
# ---------------------------------------------------------------------------

class AudioMamba(Mamba3Model):
    def __init__(self, d_state, d_model, d_inner, nheads, headdim, num_layers,
                 mimo_rank=1, cond_dim=128):
        super().__init__(d_state, d_model, d_inner, nheads, headdim, num_layers,
                         mimo_rank=mimo_rank, cond_dim=cond_dim)
        self.d_model = d_model
        self.cond_dim = cond_dim

    @property
    def num_parameters(self) -> int:
        return sum(int(p.numel()) for p in self.parameters())

    def forward(self, x, cond=None):
        for layer in self.layers:
            x = layer(x, cond=cond)
        return x


def make_model(cfg: dict) -> AudioMamba:
    return AudioMamba(
        d_state=cfg["d_state"],
        d_model=cfg["d_model"],
        d_inner=cfg["d_inner"],
        nheads=cfg["nheads"],
        headdim=cfg.get("headdim", cfg["d_inner"] // cfg["nheads"]),
        num_layers=cfg["num_layers"],
        mimo_rank=cfg.get("mimo_rank", 1),
        cond_dim=cfg.get("cond_dim", 128),
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int) -> np.ndarray:
    import torchaudio
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0).numpy().astype(np.float32)


def frame_waveform(wave: np.ndarray, seq_len: int) -> np.ndarray:
    n = len(wave)
    if n < seq_len + 1:
        wave = np.concatenate([wave, np.zeros(seq_len + 1 - n, dtype=np.float32)])
        n = len(wave)
    n_chunks = n // (seq_len + 1)
    return wave[: n_chunks * (seq_len + 1)].reshape(n_chunks, seq_len + 1)


class AudioWaveDataset(torch.utils.data.IterableDataset):
    def __init__(self, root: str, cfg: dict):
        self.root = Path(root)
        self.seq_len = cfg["seq_len"]
        self.d_model = cfg["d_model"]
        self.cond_dim = cfg.get("cond_dim", 128)
        self.target_sr = cfg.get("sample_rate", 44100)
        self.files = sorted(
            p for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
        )
        if not self.files:
            raise RuntimeError(f"No audio files found in {root}")

    def __iter__(self) -> Iterator[tuple]:
        worker_info = torch.utils.data.get_worker_info()
        files = list(self.files)
        if worker_info is not None:
            per_worker = math.ceil(len(files) / worker_info.num_workers)
            files = files[worker_info.id * per_worker:(worker_info.id + 1) * per_worker]
        else:
            random.Random().shuffle(files)

        for path in files:
            try:
                wave = load_audio(str(path), self.target_sr)
            except Exception as exc:
                print(f"[skip] {path}: {exc}")
                continue
            for chunk in frame_waveform(wave, self.seq_len):
                if self.d_model == 1:
                    inp = chunk[:-1].reshape(-1, 1)
                    tgt = chunk[1:].reshape(-1, 1)
                else:
                    inp = chunk[:-1]
                    tgt = chunk[1:]
                cond = np.zeros(self.cond_dim, dtype=np.float32)
                yield (torch.from_numpy(inp), torch.from_numpy(tgt),
                       torch.from_numpy(cond))


class SyntheticDataset(torch.utils.data.IterableDataset):
    def __init__(self, cfg: dict, num_samples: int = 1000):
        self.seq_len = cfg["seq_len"]
        self.d_model = cfg["d_model"]
        self.cond_dim = cfg.get("cond_dim", 128)
        self.num_samples = num_samples
        self.rng = np.random.RandomState(42)

    def __iter__(self) -> Iterator[tuple]:
        for _ in range(self.num_samples):
            freq = 0.02 + self.rng.rand() * 0.08
            phase = self.rng.rand() * 2 * math.pi
            t = np.arange(self.seq_len + 1, dtype=np.float32)
            chunk = np.sin(2 * math.pi * freq * t + phase)
            chunk += 0.05 * self.rng.randn(self.seq_len + 1).astype(np.float32)
            inp = chunk[:-1].reshape(-1, 1) if self.d_model == 1 else chunk[:-1]
            tgt = chunk[1:].reshape(-1, 1) if self.d_model == 1 else chunk[1:]
            cond = np.zeros(self.cond_dim, dtype=np.float32)
            yield (torch.from_numpy(inp), torch.from_numpy(tgt),
                   torch.from_numpy(cond))


def collate(batch):
    return (torch.stack([b[0] for b in batch]),
            torch.stack([b[1] for b in batch]),
            torch.stack([b[2] for b in batch]))


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict, output_dir: Path, dataset_dir: Optional[str] = None,
          device: Optional[str] = None, use_amp: bool = True,
          resume: Optional[str] = None, smoke_steps: Optional[int] = None,
          validate: bool = False):
    torch.manual_seed(cfg.get("seed", 0))
    np.random.seed(cfg.get("seed", 0))
    random.seed(cfg.get("seed", 0))

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = use_amp and device.startswith("cuda")
    dtype = torch.float16 if use_amp else torch.float32
    scaler = torch.amp.GradScaler(device.split(":")[0], enabled=use_amp)

    model = make_model(cfg).to(device)
    print(f"device={device}  amp={use_amp}")
    print(f"model params: {model.num_parameters:,}")

    if dataset_dir and Path(dataset_dir).exists():
        dataset = AudioWaveDataset(dataset_dir, cfg)
    else:
        print("[warn] no dataset-dir; using synthetic sinusoids")
        dataset = SyntheticDataset(cfg)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        num_workers=cfg.get("num_workers", 0),
        collate_fn=collate,
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        betas=tuple(cfg.get("betas", (0.9, 0.999))),
        weight_decay=cfg.get("weight_decay", 0.01),
    )

    grad_accum = max(1, int(cfg.get("grad_accum", 1)))
    criterion = torch.nn.MSELoss()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.yaml").write_text(yaml.dump(cfg))

    step = 0
    if resume:
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt.get("optimizer_state_dict", optimizer.state_dict()))
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step = ckpt.get("step", 0)
        print(f"resumed from {resume} at step {step}")

    model.train()
    start = time.time()
    done = False
    accum = 0
    for epoch in range(cfg["epochs"]):
        for x, y, cond in loader:
            x, y, cond = x.to(device), y.to(device), cond.to(device)

            with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
                pred = model(x, cond=cond)
                loss = criterion(pred, y) / grad_accum

            scaler.scale(loss).backward()
            accum += 1

            if accum % grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                accum = 0

                if step == 1 or step % cfg.get("log_every", 10) == 0:
                    print(f"step {step:>6}  loss={loss.item() * grad_accum:.6f}  "
                          f"lr={optimizer.param_groups[0]['lr']:.2e}  "
                          f"{time.time() - start:.1f}s")

                if step % cfg.get("save_every", 1000) == 0:
                    torch.save({"step": step,
                                "model_state_dict": model.state_dict(),
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scaler_state_dict": scaler.state_dict(),
                                "config": cfg},
                               output_dir / f"checkpoint_{step:07d}.pt")

                if smoke_steps is not None and step >= smoke_steps:
                    done = True
                    break
        if done:
            break

    torch.save({"model_state_dict": model.state_dict(), "config": cfg},
               output_dir / "final.pt")

    # Export plugin weights
    b = model.layers[0]
    bin_path = output_dir / "plugin_weights.bin"
    export_model(model, str(bin_path), cond_dim=cfg.get("cond_dim", 128), random_init=False)
    generate_config_h(b.d_state, b.d_model, b.d_inner, b.nheads, b.headdim,
                      len(model.layers), cfg.get("cond_dim", 128), b.mimo_rank,
                      str(output_dir / "config.h"))
    print(f"done. exported {bin_path} and {output_dir / 'config.h'}")

    if validate:
        import subprocess
        validate_script = os.path.join(_SCRIPTS_DIR, "validate_export.py")
        print(f"$ {sys.executable} {validate_script} {bin_path}")
        subprocess.run([sys.executable, validate_script, str(bin_path)], check=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Train Mamba-3 plugin model")
    p.add_argument("--config", default="training/configs/laptop_rtx4070.yaml")
    p.add_argument("--dataset-dir", default=None)
    p.add_argument("--output-dir", default="runs/run1")
    p.add_argument("--device", default=None)
    p.add_argument("--resume", default=None)
    p.add_argument("--smoke-test", action="store_true", help="short synthetic run")
    p.add_argument("--steps", type=int, default=5, help="steps for --smoke-test")
    p.add_argument("--validate", action="store_true", help="run validate_export.py after export")

    # Frequent overrides
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)

    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    for flag, key in [("batch_size", "batch_size"), ("grad_accum", "grad_accum"),
                      ("num_workers", "num_workers"), ("lr", "lr"),
                      ("epochs", "epochs"), ("seq_len", "seq_len")]:
        val = getattr(args, flag.replace("_", "_"))
        if val is not None:
            cfg[key] = val
    if args.no_amp:
        cfg["amp"] = False

    if args.smoke_test:
        cfg["epochs"] = 1
        if args.dataset_dir is None:
            print("[smoke] using synthetic data")

    train(cfg,
          output_dir=Path(args.output_dir),
          dataset_dir=args.dataset_dir,
          device=args.device,
          use_amp=not args.no_amp,
          resume=args.resume,
          smoke_steps=args.steps if args.smoke_test else None,
          validate=args.validate)


if __name__ == "__main__":
    main()
