#!/usr/bin/env python3
"""
training/losses.py

Task-matched losses for training audio effects and restoration models.

Why this file exists
--------------------
`train.py` originally optimised plain MSE between a waveform and its one-sample
shift. That objective is a poor proxy for how an effect is judged: it does not
weight the spectrum, it does not respect level, and it is dominated by whatever
the loudest band happens to be. The losses here are the standard ones for this
job and are the ones the project's own training guidance recommends:

  * time-domain L1 with pre-emphasis  - broadband waveform fidelity, with the
    high end protected (SI-SDR alone ignores frequency weighting)
  * SI-SDR (scale-invariant)          - restoration/denoising objective that does
    not reward simply making the output quieter or louder
  * multi-resolution STFT (MRSTFT)    - spectral convergence + log-magnitude L1
    across several window sizes; the standard perceptual stand-in for "tone"

Known limitations (documented rather than hidden - see ARCHITECTURE_FIDELITY.md):
  * SI-SDR is bounded by reference SNR when references are noisy, is ill-defined
    for silent targets, and can correlate negatively with perceived noisiness, so
    it is combined with a spectral term and never used as the sole evidence.
  * MRSTFT phase is ignored (magnitude only), which is why the time-domain term
    stays in the mix.
  * None of these replace listening tests on level-matched material.

Everything uses torch core ops only - no torchaudio, no auraloss dependency.

Self-test (no data needed):

    python3 training/losses.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Time domain
# ---------------------------------------------------------------------------

def pre_emphasis(x: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
    """First-order high-pass y[n] = x[n] - coeff * x[n-1]. Shape [B, 1, T]."""
    if coeff <= 0.0:
        return x
    x_prev = F.pad(x[:, :, :-1], (1, 0))
    return x - coeff * x_prev


class PreEmphL1Loss(nn.Module):
    """L1 on the pre-emphasised waveform (Wright & Valimaki, 2019 lineage)."""

    def __init__(self, coeff: float = 0.97):
        super().__init__()
        self.coeff = coeff

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(pre_emphasis(pred, self.coeff), pre_emphasis(target, self.coeff))


# ---------------------------------------------------------------------------
# SI-SDR
# ---------------------------------------------------------------------------

def si_sdr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Scale-invariant signal-to-distortion ratio in dB, per item. Shape [B, 1, T].

    Returns a tensor of shape [B] holding 10*log10(||s_target||^2 / ||e_noise||^2).
    Items whose target is (near) silent are marked with NaN so the caller can
    substitute a waveform loss instead of producing a meaningless dB value.
    """
    pred = pred.reshape(pred.shape[0], -1)
    target = target.reshape(target.shape[0], -1)

    tgt_energy = (target ** 2).sum(dim=-1)
    silent = tgt_energy < eps * 100  # ill-defined case: nothing to separate

    scale = (pred * target).sum(dim=-1) / (tgt_energy + eps)          # alpha
    s_target = scale.unsqueeze(-1) * target
    e_noise = pred - s_target

    ratio = (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps)
    out = 10.0 * torch.log10(ratio + eps)
    return torch.where(silent, torch.full_like(out, float("nan")), out)


class SISDRLoss(nn.Module):
    """
    Negative SI-SDR in dB, mean-reduced over items.

    Silent-target items fall back to L1 on that item so the batch always yields a
    finite, differentiable value.
    """

    def __init__(self, fallback_weight: float = 1.0):
        super().__init__()
        self.fallback_weight = fallback_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        vals = si_sdr(pred, target)
        valid = ~torch.isnan(vals)
        if valid.any():
            term = -vals[valid].mean() / 20.0      # ~O(1) scale for mixing
        else:
            term = pred.new_zeros(())
        if not valid.all():
            fb = F.l1_loss(pred[~valid], target[~valid])
            term = term + self.fallback_weight * fb
        return term


# ---------------------------------------------------------------------------
# Multi-resolution STFT
# ---------------------------------------------------------------------------

class MultiResSTFTLoss(nn.Module):
    """
    Sum over resolutions of (spectral convergence + log-magnitude L1).

    Mirrors the commonly used formulation (Yamamoto et al., 2019; Parallel
    WaveGAN; auraloss.freq.MultiResolutionSTFTLoss) using torch.stft only.
    """

    def __init__(self,
                 fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512),
                 win_lengths=(512, 1024, 2048)):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.fft_sizes = list(fft_sizes)
        self.hop_sizes = list(hop_sizes)
        self.win_lengths = list(win_lengths)
        self.register_buffer("_ones", torch.ones(1))  # keeps .to(device) valid

    @staticmethod
    def _to_2d(x: torch.Tensor) -> torch.Tensor:
        """Accept [B, T], [B, 1, T] or [B, T, 1] and return something stft-ready."""
        if x.dim() == 2:
            return x
        if x.dim() == 3:
            if x.shape[1] == 1:          # [B, 1, T]
                return x[:, 0, :]
            if x.shape[2] == 1:          # [B, T, 1]
                return x[:, :, 0]
            b, c, t = x.shape            # multi-channel: fold channels into batch
            return x.reshape(b * c, t)
        raise ValueError(f"expected 2D or 3D audio tensor, got shape {tuple(x.shape)}")

    def _mag(self, x: torch.Tensor, n_fft: int, hop: int, win: int) -> torch.Tensor:
        window = torch.hann_window(win, device=x.device, dtype=x.dtype)
        spec = torch.stft(x, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=window, center=True, return_complex=True)
        return spec.abs().clamp_min(1e-7)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # torch.stft wants [B, T]; the model works on [B, T, 1].
        p = self._to_2d(pred)
        t = self._to_2d(target)

        total = pred.new_zeros(())
        for n_fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            mp = self._mag(p, n_fft, hop, win)
            mt = self._mag(t, n_fft, hop, win)
            # spectral convergence
            sc = torch.linalg.vector_norm(mt - mp, dim=(-2, -1)) / \
                 (torch.linalg.vector_norm(mt, dim=(-2, -1)) + EPS)
            # log-magnitude L1
            lm = F.l1_loss(torch.log(mp), torch.log(mt))
            total = total + sc.mean() + lm
        return total / len(self.fft_sizes)


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

class CompositeLoss(nn.Module):
    """weighted combination of pre-emph L1, SI-SDR and MRSTFT."""

    def __init__(self, time_weight=1.0, sisdr_weight=1.0, stft_weight=1.0,
                 preemph=0.97, fft_sizes=(512, 1024, 2048),
                 hop_sizes=(128, 256, 512), win_lengths=(512, 1024, 2048)):
        super().__init__()
        self.w_time = float(time_weight)
        self.w_sisdr = float(sisdr_weight)
        self.w_stft = float(stft_weight)
        self.time = PreEmphL1Loss(preemph)
        self.sisdr = SISDRLoss()
        self.stft = MultiResSTFTLoss(fft_sizes, hop_sizes, win_lengths)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = pred.new_zeros(())
        if self.w_time:
            loss = loss + self.w_time * self.time(pred, target)
        if self.w_sisdr:
            loss = loss + self.w_sisdr * self.sisdr(pred, target)
        if self.w_stft:
            loss = loss + self.w_stft * self.stft(pred, target)
        return loss

    def components(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Per-term values, for logging. Never used in the backward pass."""
        with torch.no_grad():
            vals = si_sdr(pred, target)
            return {
                "time": float(self.time(pred, target)),
                "sisdr_loss": float(self.sisdr(pred, target)),
                "sisdr_db": float(torch.nanmean(vals)) if (~torch.isnan(vals)).any()
                            else float("nan"),
                "stft": float(self.stft(pred, target)),
            }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_LOSS_NAMES = ("mse", "l1", "sisdr", "mrstft", "hybrid")


def build_loss(name: str, cfg: dict | None = None, device: str = "cpu") -> nn.Module:
    """
    Build a loss by name.

    Config keys (all optional):
        loss_weights: {time, sisdr, stft}  - weights for 'hybrid'
        loss_preemph: float                - pre-emphasis coefficient (0 disables)
        loss_fft_sizes / loss_hop_sizes / loss_win_lengths: lists of ints
    """
    cfg = cfg or {}
    name = (name or "mse").lower()
    if name not in _LOSS_NAMES:
        raise ValueError(f"unknown loss '{name}' (expected one of {_LOSS_NAMES})")

    if name == "mse":
        return nn.MSELoss()
    if name == "l1":
        return PreEmphL1Loss(cfg.get("loss_preemph", 0.97))
    if name == "sisdr":
        return SISDRLoss()
    if name == "mrstft":
        return MultiResSTFTLoss(
            cfg.get("loss_fft_sizes", (512, 1024, 2048)),
            cfg.get("loss_hop_sizes", (128, 256, 512)),
            cfg.get("loss_win_lengths", (512, 1024, 2048)),
        )

    weights = cfg.get("loss_weights", {}) or {}
    return CompositeLoss(
        time_weight=weights.get("time", 1.0),
        sisdr_weight=weights.get("sisdr", 1.0),
        stft_weight=weights.get("stft", 1.0),
        preemph=cfg.get("loss_preemph", 0.97),
        fft_sizes=cfg.get("loss_fft_sizes", (512, 1024, 2048)),
        hop_sizes=cfg.get("loss_hop_sizes", (128, 256, 512)),
        win_lengths=cfg.get("loss_win_lengths", (512, 1024, 2048)),
    ).to(device)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def _self_test() -> int:
    torch.manual_seed(0)
    failures = []

    def check(name, cond, detail=""):
        status = "ok  " if cond else "FAIL"
        print(f"  [{status}] {name} {detail}")
        if not cond:
            failures.append(name)

    print("losses.py self-test")

    T = 8192
    t = torch.arange(T, dtype=torch.float32)
    clean = torch.sin(2 * math.pi * 440.0 * t / 48000.0).reshape(1, 1, -1)
    noise = (0.1 * torch.randn(1, 1, T))
    noisy = clean + noise

    # --- SI-SDR properties -------------------------------------------------
    same = si_sdr(clean, clean)
    check("si_sdr(x, x) is large and positive", float(same) > 60.0, f"({float(same):.1f} dB)")
    # Against a CLEAN reference, a cleaner estimate must score higher.
    near = float(si_sdr(0.95 * clean + 0.05 * noise, clean))
    clean_ref = float(si_sdr(clean, clean))
    far = float(si_sdr(noise, clean))
    check("si_sdr rewards the closer estimate", clean_ref > near > far,
          f"(exact={clean_ref:.1f} > near={near:.1f} > noise={far:.1f} dB)")
    # Use an imperfect estimate: for an exact match the ratio is 0/0 up to
    # float32 rounding, so the raw dB value is numerically meaningless.
    est = 0.95 * clean + 0.05 * noise
    scaled = float(si_sdr(3.0 * est, clean))
    check("si_sdr is scale invariant", abs(scaled - near) < 0.5,
          f"(1x={near:.2f} vs 3x={scaled:.2f})")
    # Documented limitation, asserted so it cannot surprise us later: with a
    # NOISY reference, an estimate that reproduces the reference noise scores
    # higher than a clean one (arXiv:2508.14623). This is why SI-SDR is never
    # used alone here.
    noisy_ref_clean = float(si_sdr(clean, noisy))
    noisy_ref_noisy = float(si_sdr(noisy, noisy))
    check("documents the noisy-reference caveat", noisy_ref_noisy > noisy_ref_clean,
          f"(reproducing reference noise={noisy_ref_noisy:.1f} dB vs clean={noisy_ref_clean:.1f} dB)")
    silent = si_sdr(noise, torch.zeros_like(noise))
    check("si_sdr flags silent targets as NaN", bool(torch.isnan(silent).all()))
    check("SISDRLoss stays finite on silent targets",
          bool(torch.isfinite(SISDRLoss()(noise, torch.zeros_like(noise)))))

    # --- MRSTFT ------------------------------------------------------------
    mrstft = MultiResSTFTLoss(fft_sizes=(256, 512), hop_sizes=(64, 128),
                              win_lengths=(256, 512))
    identical = float(mrstft(clean, clean))
    different = float(mrstft(noise, clean))
    check("mrstft(x, x) is ~0", identical < 1e-3, f"({identical:.2e})")
    check("mrstft separates noise from tone", different > identical + 1.0,
          f"({different:.3f} vs {identical:.2e})")

    # --- gradients flow ----------------------------------------------------
    for name in _LOSS_NAMES:
        loss_fn = build_loss(name, {}, "cpu")
        p = noisy.clone().requires_grad_(True)
        val = loss_fn(p, clean)
        val.backward()
        check(f"{name}: finite value and gradient",
              bool(torch.isfinite(val)) and p.grad is not None and
              bool(torch.isfinite(p.grad).all()), f"(value={float(val.detach()):.4f})")

    # --- registered config path -------------------------------------------
    composite = build_loss("hybrid", {"loss_weights": {"time": 2.0, "sisdr": 0.5, "stft": 1.5},
                                      "loss_preemph": 0.9}, "cpu")
    check("hybrid honours config weights",
          composite.w_time == 2.0 and composite.w_sisdr == 0.5 and composite.w_stft == 1.5)
    comps = composite.components(noisy, clean)
    check("components() reports every term",
          set(comps) == {"time", "sisdr_loss", "sisdr_db", "stft"},
          f"({', '.join(f'{k}={v:.3f}' for k, v in comps.items())})")

    print()
    if failures:
        print(f"SELF-TEST FAILED: {len(failures)} check(s): {', '.join(failures)}")
        return 1
    print("SELF-TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
