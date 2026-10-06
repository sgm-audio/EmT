# Verification Report — "Mamba-3 + custom LoRA inside a plugin" career strategy

**Date:** 2026-10-04
**Method:** primary-source research (papers, official docs, price pages, job posts, court reporting), independent inspection of the `EmT` repository this strategy is written for, and an independently built and executed run of the repo's own SSM benchmark on this machine.

**Bottom line:** the business logic is largely sound and several load-bearing claims check out (real-time budgets, GPU rental costs, data-rights risk, loudness-matching in demos, JUCE demand). But the *technical differentiator* the strategy is anchored to is not currently what the text says it is: the repo does not implement the Mamba-3 paper's recurrence, the complex-valued state is inactive in the shipped C++ path, MIMO is unused, and **there is no LoRA implementation anywhere in the repo** — only a GUI mockup package. Verdict: keep the strategy, fix the identity of the "differentiator," and correct five specific factual/economic claims before publishing.

---

## 0. Post-review closure addendum (2026-10-04, second pass)

The technical findings below were correct when written and are kept as the
historical record.  Four of the six register deviations were then closed by code
changes (D3, D4, D5, D6); D1/D2 remain open.  Current state is machine-checked by
`python3 scripts/verify_deviations.py`, and the full ledger lives in
`ARCHITECTURE_FIDELITY.md` §2 / §4c:

| Deviation | Finding in this report | Now |
|---|---|---|
| D3 — MIMO unused | §1 table "MIMO … Unused" | **CLOSED** — rank-R files are refused at load (`-16`), export and config validation; the rank-R *implementation* remains unimplemented and is scoped in `Prompts/DEVIATION_RESOLUTION/D3_MIMO_RANK.md` |
| D4 — input-derived `A` | §1 table "Deviation" | **CLOSED** — documented as a project variant; §1.2 of the register forbids describing it as the paper's parameterization |
| D5 — periodic state renormalization | §1 table "Deviation / stability workaround"; appendix row 9 | **CLOSED** — the guard is removed from C++ and both references; stability now follows from the A-stable update, and a continuity + long-run test guards it.  The appendix's "characterised, not fixed" note below is superseded |
| D6 — `dt_film_proj_w` unused | not covered by this report | **CLOSED** — Δ-FiLM is applied in the C++, NumPy and PyTorch forward passes with one convention, only when conditioning is active |
| D1 — no exponential-trapezoidal recurrence | §1 table "Not reproduced" | **OPEN** — prompt: `Prompts/DEVIATION_RESOLUTION/D1_D2_TRAPEZOIDAL_COMPLEX_STATE.md` |
| D2 — complex state inactive | §1 table "Inactive" | **OPEN** — same prompt; the claim policy in `ARCHITECTURE_FIDELITY.md` §1.2 still applies ("Mamba-3-inspired" until this lands) |

The LoRA finding (no implementation; GUI mockup only) is unchanged and still
binds: `ARCHITECTURE_FIDELITY.md` §1.2 forbids "runtime LoRA adapters".

---

## 1. Verdict summary

| # | Claim in the text | Verdict | Basis |
|---|---|---|---|
| 1 | Freelance/indie work pays for real-time plugins; JUCE is the default (VST3, AU, AAX, CLAP) | **Mostly confirmed, one correction** | JUCE is the de-facto standard and appears in contract listings; CLAP is *not* natively supported by JUCE |
| 2 | Solid DSP, UI, presets, installers, support is what pays | **Confirmed (qualitative)** | Matches custom-plugin market ($30k–100k builds, 15–25 %/yr maintenance) |
| 3 | Neural features only inside the real-time CPU/latency budget | **Confirmed** | RNNoise ≈2.5 % core @48 kHz; NAM 10–30 % per instance; RTNeural designed for this constraint |
| 4 | "Mamba and Mamba-2 are the public foundation; treat Mamba-3 as whatever paper/code you can cite" | **Outdated — now fixable** | Mamba-3 is real: arXiv 2603.15569 (ICLR 2026), code released 2026-03-17 in `state-spaces/mamba`, still maintained (fix 2026-07-22) |
| 5 | The repo implements "Mamba-3" | **Not verified — contradicts the paper** | C++ uses a 2-term Tustin recurrence; the paper's exponential-trapezoidal rule is a 3-term recurrence; `ssm_imag` is never written; MIMO unused; `trap` weights unused |
| 6 | Small causal SSM can run real time if state/hidden size are tiny | **Confirmed by measurement** | 1 layer ≈2.7 µs/sample; 2 layers ≈5.4 µs/sample; 6 layers 17.7–23.9 µs/sample (**fails** the repo's own 10 µs budget) on this 2-vCPU sandbox |
| 7 | "Under 10 % of one core" CPU budget | **Too optimistic as stated** | 1 layer ≈9 % core (repo's 1.79 µs figure) to ≈13 % (measured here); 2 layers ≈18–26 %; excludes plugin glue |
| 8 | LoRA: freeze base, train A/B, merge for shipping, runtime only if many adapters needed | **Confirmed from the LoRA paper** | Merge W₀+BA ⇒ no added inference latency; unmerged multi-task serving is a documented limitation |
| 9 | "Custom LoRA inside a plugin" is a research differentiator | **Unsupported and, in this repo, unimplemented** | LoRA-for-audio precedent exists in *generation* (Stable Audio 3, TTS, ggml merge), not real-time DSP; repo has zero LoRA code |
| 10 | RunPod: rent a single 24 GB GPU to train | **Confirmed, cheap** | 4090 ≈$0.34/hr (Community) / $0.74/hr (Secure); A5000 from ≈$0.16–0.27/hr; per-second billing, no egress fees |
| 11 | Only train on data you have rights to; unlicensed commercial records are a product/legal problem | **Confirmed, and stronger than stated** | RIAA v. Suno/Udio; Warner settled with licensing deals; exposure up to $150k/work and $9B+ if expanded; MUSDB18 is explicitly non-commercial |
| 12 | Task-matched losses (SI-SDR / waveform L1; multi-resolution spectral) | **Confirmed as standard, with caveats** | SI-SDR is bounded by reference SNR / ill-defined for silent targets; MRSTFT (auraloss) is standard for tone |
| 13 | Distill to a tiny student; fp16 **storage**, fp32 math | **Confirmed** | Distillation results in real-time speech SSMs; fp16 CPU *compute* is frequently far slower than fp32 |
| 14 | Export weights + JSON ranges; plugin loads with no Python | **Confirmed as standard practice** | RTNeural JSON, NAM `.nam`, repo's flat `.bin` + generated `config.h` |
| 15 | Reliable latency reporting / prefer zero look-ahead | **Confirmed** | CLAP/JUCE both expose latency reporting; but Mamba-1/2's causal conv is already streamable with zero look-ahead, so *removing* it saves compute, not latency |
| 16 | Grok will not train your model, host your GPU, or get you hired | **Confirmed** | xAI offers inference only (fine-tuning "Not available"); no GPU hosting |
| 17 | Grok "cuts weeks off boilerplate" | **Partially supported, task-dependent** | GitHub: 55 % faster on a bounded task; METR RCT: **19 % slower** on familiar mature repos (2025), follow-up ≈18 % faster with wide CI (2026); McKinsey: 46 % faster routine, <10 % complex |
| 18 | Before/after demos must be gain-matched | **Confirmed** | 1–2 dB is enough to flip preference; level matching is the standard control |
| 19 | "$2k–8k for a defined prototype" | **Plausible, low anchor** | 30–160 h at $50–70/hr contract rates; agency-scale plugin builds quote $30k+, no-code platforms anchor far lower |
| 20 | "Owning a small plugin often beats low freelance rates if the demo is good" | **Unsupported for a beginner** | Documented cases: ≈$190 early revenue; £2,000/18 months; a creator review drove £2,500 in 2 days → distribution, not demo quality, dominates |
| 21 | 90-day sequence reaches paid outcomes | **Directionally right, timeline optimistic** | Artifacts are achievable in 90 days; outbound conversion in a relationship-driven niche typically takes longer |
| 22 | Social: post the sound, one demo a week, one link hub | **Reasonable heuristics, unverifiable as stated** | Sound-first distribution is consistent with NAM/TONE3000 dynamics; cadence advice has no direct evidence |

---

## 2. The model-name question — and a problem in the repo

**What is true now (October 2026):** Mamba-3 exists and is citable and runnable.

- Paper: *Mamba-3: Improved Sequence Modeling using State Space Principles*, arXiv:2603.15569, submitted 2026-03-16, **ICLR 2026** (oral).
- Code: released in `state-spaces/mamba` on **2026-03-17** ("Mamba-3 code release"), with continued maintenance (ROCm support; a Blackwell-kernel correctness fix on 2026-07-22). Mamba-3 (SISO and MIMO) kernels are present.
- The paper's three contributions: **exponential-trapezoidal discretization** (a **three-term** recurrence `h_t = α_t h_{t−1} + β_t B_{t−1} x_{t−1} + γ_t B_t x_t`), a **complex-valued state update** implemented via a data-dependent RoPE trick, and a **MIMO** formulation.

So the text's advice — "treat Mamba-3 as whatever paper and code you can actually cite and reproduce" — was prudent but is now answerable: **cite arXiv:2603.15569 and `state-spaces/mamba`, and then actually reproduce it.** That last part is where this repository currently fails.

**Repo inspection (`src/core/ssm_block.cpp`, `src/core/ssm_block.h`, `scripts/reference_mamba3.py`):**

| Mamba-3 paper | EmT implementation | Status |
|---|---|---|
| Exponential-trapezoidal, 3-term recurrence with `B_{t−1}x_{t−1}` history term | 2-term recurrence `h ← Ā h + B̄·(B_t x_t)` with Tustin/bilinear `Ā=(1+Δt·A/2)/(1−Δt·A/2)`, `B̄=Δt/(1−Δt·A/2)` | **Not reproduced** (classical trapezoidal rule, not the paper's exponential-trapezoidal update) |
| Complex-valued state (2×2 rotation blocks / RoPE) | `ssm_imag[]` is allocated but **never written in the recurrence** | **Inactive** (D2; the periodic normalization that used to scale it was removed for D5) |
| MIMO (rank-R B/C coupling) | Weights loaded/exported, but the recurrence indexes rank 0 only | **Unused** — since the D3 closure, rank > 1 is refused at load instead of silently downgraded |
| Input-dependent selective Δ, B, C (A a learned per-head parameter) | Δ and A are both derived from input projections (`A_t = −softplus(dd_A)`, clamped at −1) | **Deviation** (input-dependent A is a legitimate variant, but not the paper's parameterization) |
| No conv1d | No conv1d | **Matches** |
| RoPE on B/C + RMSNorm on B/C + learnable biases | Implemented (cumulative angle state, RMSNorm, B/C biases) | **Matches** |
| State stability from parameterization | Manual **periodic state renormalization every 4096 samples** (absent from the paper) | **Removed** (D5 closure, 2026-10-04); stability now follows from the A-stable discretization |

**Consequence for the strategy:** the "Mamba-3" label is currently doing more work than the code. For a technical audience — exactly the audience that refers contract work — claiming Mamba-3 while shipping a Tustin-approximated real-state SISO SSM is a credibility risk. Either bring the recurrence to the paper's form (3-term exponential-trapezoidal + active complex state + rank-0→rank-R path) or rename the architecture honestly (e.g., "Mamba-3-inspired causal SSM") in the README, docs, and demo copy.

**Also correct this:** the README claims removing Mamba-1/2's causal conv yields "0 samples of algorithmic delay." Causal `conv1d` in streaming implementations is stateful with **no look-ahead** (`causal_conv1d_update` in `state-spaces/mamba`); removing it is a compute/simplification win, not a latency win. The "zero latency" differentiator is marketing, not engineering.

---

## 3. Real-time budget: measured, not asserted

I compiled the repo's own benchmark without CMake (CMake is unavailable in this sandbox — see appendix) and ran it:

- **1 layer:** 2.66–2.77 µs/sample → **≈13 % of one core at 48 kHz** (this machine)
- **2 layers:** 5.33–5.67 µs/sample → **≈26 % of one core at 48 kHz**
- **6 layers:** 17.7–23.9 µs/sample → **fails**; the harness reports `Configurations meeting < 10.0 µs/sample budget: 24 / 36`
- Block size (64→512) makes almost no difference: the block path is a per-sample loop (`process_block` falls back to `process_sample`, documented in the source).
- The harness's own "real-time budget" threshold is **10 µs/sample = 48 % of one core at 48 kHz**, not "under 10 % of one core."
- The benchmark config is *tiny* (d_state 16, d_model 16, d_inner 32, 2 heads, headdim 16). The compiled maxima in `config.h` are much larger (d_state 32, d_model 256, d_inner 512, 8 heads). So these are favorable numbers.
- Repo docs claim 1.79 µs/sample (1 layer) / 3.73 µs (2 layers); I measured ≈1.5× slower on this shared 2-vCPU sandbox. Machine-dependent — **measure on the target laptop, at 48 and 96 kHz**, exactly as the text advises.

**Implication:** the text's "for example under 10 % of one core" target is met only by a single small layer on a fast core. Anything you sell must quote a measured number on named hardware; the current repo number ("1.79 µs/sample") is incompatible with the repo's own 10 µs *threshold* when layers ≥ 3, and the 6-layer configuration is already outside budget.

**Comparators (so the number means something):** RNNoise runs at ≈2.5 % of one core at 48 kHz mono with ≈10 ms latency; NAM is 10–30 % per instance (heavier models; "feather" variants much lighter); a distilled 1-layer streaming Mamba speech-enhancer reports 25 ms latency and RTF 0.11. The EmT core sits in a believable range for a *single* small layer, and nowhere near "free."

---

## 4. LoRA: mechanism confirmed, differentiator not

- **LoRA mechanics are as the text describes.** From arXiv:2106.09685: freeze W₀, train ΔW = BA with r ≪ min(d,k); trainable parameters r(d+k); at deployment compute W = W₀ + BA, which by construction adds **no inference latency**; the paper explicitly notes it is *not* straightforward to batch different tasks with different (A,B) in one pass when merged.
- **Therefore "runtime LoRA" is only worth it when one binary must host many adapters** — the text says exactly this, and it is the correct call. Merged weights are the simpler, faster ship path. For the model sizes here (≈160 K–3.4 M params per `TRAINING_PLAN.md`), full per-preset weight files are small; the LoRA machinery buys little.
- **Precedent check:** LoRA in audio exists, but in **generation**, not real-time DSP — e.g., Stable Audio 3 LoRA training studios, GRPO-trained style LoRAs for TTS, and a C++/ggml music model that merges LoRA/LoKr into weights at load. I found no evidence of a shipped real-time DSP plugin whose selling point is customer-trained LoRA adapters.
- **Repo reality:** `grep -ri lora` finds **no implementation** — only `packages/emt_lora_matrix_companion/` (PNG mockups and a GUI spec) plus an asset zip. There is no adapter training code, no adapter export format, no runtime adapter path, and no CLI parameter for adapters.

**Verdict:** "Mamba-3 + custom LoRA inside a plugin" is currently a **roadmap item and a UI mockup**, not a differentiator you can demo. If the strategy depends on it, build it before writing the post. Treat it as an *engineering* differentiator at best (fast iteration on per-client tone), not a research claim — the paper novelty belongs to the Mamba-3 authors, and LoRA belongs to Hu et al.

---

## 5. Training pipeline and the "loss that matches the product" gap

The text's training advice is conventional and correct (causal/streaming training, task-matched losses, distillation, quantization, export). Two verifications and one repo gap:

- **SI-SDR** is standard (Le Roux et al. 2018; `auraloss.time.SISDRLoss`) but has documented failure modes: it is **bounded by the reference SNR** when references are noisy, is **ill-defined for zero-energy (silent) targets**, does not weight frequency regions, and can be **negatively correlated with perceived noisiness** across models. Pair it with a spectral term and with listening tests.
- **Multi-resolution STFT loss** is standard for tone/perceptual alignment (Yamamoto et al. 2019; Parallel WaveGAN; `auraloss.freq.MultiResolutionSTFTLoss`).
- **Distillation to a 1-layer streaming model** is demonstrated for SSM audio (RT-SEMamba: 8-layer teacher → 1-layer student, 3.18 PESQ, fixed 25 ms latency, RTF 0.11, 2.64× speedup), which supports the text's "distill until it fits" instruction.
- **fp16 caution confirmed:** on CPUs, fp16/bf16 *compute* is frequently much slower than fp32 (PyTorch/SBERT CPU bench ≈0.17×; one reported case ≈26× slower; CMSIS-NN fp16 conv slower than fp32 below 4 input channels). Use fp16 for **storage**, fp32 for math — as the text says — and measure.
- **Repo gap (now closed — see Appendix C):** `training/train.py` used `torch.nn.MSELoss()` on next-sample waveform prediction. That is neither of the losses the text recommends (no SI-SDR/L1, no multi-resolution spectral, no loudness term) and is a weak objective for a perceptually-motivated effect. At the time of writing the document's own advice was not implemented in the code it refers to; [Appendix C](#appendix-c--remediation-log-2026-10-04) records the fix.

---

## 6. Data rights and legal risk

The text's warning is correct and, if anything, understated:

- **RIAA v. Suno / Udio** (filed 2024-06-24) allege unlicensed training on copyrighted sound recordings, with statutory exposure up to **$150,000 per work**. Warner settled with Suno (Nov 2025, licensing partnership + Songkick) and with Udio (Oct 2025); UMG/Sony settled with Udio; **Sony/UMG continue against Suno**, moving in 2026 to expand the claim from 560 to **61,026 works**, which would push exposure past **$9B**. The litigation is now centered on **training-data provenance** because both defendants effectively conceded the corpora contained copyrighted recordings.
- **The most obvious public dataset is not usable commercially:** MUSDB18/MUSDB18-HQ is "for educational purposes only … should not be used for any commercial purpose without the express permission of the copyright holders" (CC BY-NC-SA components). Slakh2100 is CC BY 4.0 (usable with attribution). The text's "your recordings, commissioned audio, public-domain, or explicitly licensed sets" is therefore not conservative advice — it is the minimum.
- **Omitted from the text but material:** client audio sent for a "private adapter" service must be licensed to the client (and to you for processing), scope/revisions/licence terms must be in writing, and outputs must avoid trademark/artist-name claims unless licensed.

---

## 7. Market, pricing, and "owning a plugin beats freelancing"

- **JUCE demand is real.** Representative contract listing: Kush Audio, "Audio Plugin Developer — Contract (C++ / JUCE)," $50–70/hr (Glassdoor). US "audio programmer" salary medians ≈$80–86k (ZipRecruiter/Glassdoor). The Audio Programmer operates a specialist audio-software recruitment business — a signal that the niche has enough volume to support dedicated hiring.
- **Custom plugin economics:** quoted engagements commonly land at **$30k–100k** depending on complexity, plus **15–25 %/yr** ongoing maintenance (market guide, 2026). No-code platforms anchor much lower (fixed-fee ≈$1,000 or 20–30 % revenue share). A **$2k–8k prototype** is consistent with 30–160 h at contract rates — plausible, but it sits at the very bottom of the market's price ladder and invites scope creep unless the deliverable is defined in hours, revisions, and IP/licence terms.
- **Owning a plugin:** documented solo outcomes are sobering — ≈**$190** aggregate early revenue for one release; **£2,000 in 18 months** for another; a **100k-subscriber creator review produced £2,500 in 2 days**. The KVR folklore number (a successful VSTi ≈200–300 copies/yr) is old but directionally consistent. The variable that moves revenue is **distribution**, not demo quality alone. "Owning a small plugin often beats low freelance rates if the demo is good" is the weakest economic claim in the text: for a first product, expected value favors freelancing while you build reach.
- **Format reality vs. this repo:** these buyers live in VST3/AU/AAX. JUCE targets all of them; **CLAP is not natively supported by JUCE** (it requires the unofficial `clap-juce-extensions`) and is hosted mainly by Bitwig, REAPER and FL Studio — not Ableton, Logic, or Pro Tools. `clap-wrapper` can wrap CLAP into other formats. The EmT repo builds **CLAP only** with a hand-rolled Win32/X11 GUI; its own audit report advertises avoiding JUCE. That choice shrinks the addressable market for a sellable plugin, which cuts against the strategy's own conclusion. If the goal is contract work, JUCE skills are what listings ask for — keep the SSM as the hook, not the framework.
- **Legal/tooling costs omitted from the text (small but real):** JUCE licence tiers (Personal free; Starter ≤$20k free; Indie ≈$50/mo or ≈$1,000 perpetual ≤$200k revenue; Pro ≈$175/mo or ≈$3,500 perpetual — per the JUCE 8 EULA discussion); **AAX requires Avid developer approval and PACE code-signing** (a 2026 writeup reports Avid covering the signing tools, an iLok dongle required, and no Pro Tools licence needed for development builds); Apple notarization ≈**$99/yr**. These are gates, not just line items.

---

## 8. The AI-assistant claims

- **"Grok will not train your model, host your GPU, or get you hired" — verified trivially.** xAI's Grok API is inference-only; its model table lists **"Fine-Tuning: Not available."** No GPU hosting. Hiring is unaffected by tool choice.
- **"Shorter loop between idea and working code" — partially supported, strongly task-dependent.** GitHub's controlled task study reported **55 % faster** completion; METR's RCT with 16 experienced open-source developers on real issues in mature repos found they were **19 % slower** with AI tools in 2025 while *believing* they were ~20 % faster; a 2026 follow-up estimates ≈**18 % faster** for a subset of the same cohort with a wide confidence interval (−38 % to +9 %). McKinsey reports ≈46 % time savings on routine work and **<10 % on complex work**. Net: the text's own guardrails ("not an oracle," "review, don't paste," "force it to be concrete," "benchmark") are exactly right, and the biggest documented failure mode is *perceived* speed without measured speed.
- **Local evidence of that failure mode:** this repo contains extensive AI-agent scaffolding and an "Executive Audit" scoring the project **96/100 — Production Candidate** (both since archived under `docs/archive/`), while its own gap analysis lists ARA components that render silence, a per-sample `process_block` stub, and MIMO/conditioning gaps — and the independent benchmark above shows the 6-layer configuration failing the project's own budget. Agentic self-audits inflate; the benchmark does not. Also note repo hygiene issues that undercut "proof by artifact": the README badge says 64/64 tests while `MIGRATION_TO_MAMBA3.md` and `test_ssm_output.txt` indicate 40/40 and 9 tests respectively, and the test log still contains the old `C:\Users\scott\OneDrive\...` path.

---

## 9. Overall logic — what holds, what breaks, what is missing

**The spine of the argument holds.** The chain *skills → hearable artifact → public proof → trust → paid work* is consistent with how this market actually hires and buys: JUCE jobs are real, neural audio ships only inside tight budgets (RNNoise/NAM/RTNeural evidence), GPU rental is trivially cheap, unlicensed data is a genuine legal hazard, demos must be loudness-matched, and "finish the plugin" is the skill companies pay for. The failure analysis ("fails if you optimize for model novelty") is well-supported by every comparator examined.

**Three load-bearing weaknesses:**

1. **The differentiator is misidentified.** The text asks you to treat "Mamba-3 + custom LoRA inside a plugin" as a research differentiator. The repo does not reproduce Mamba-3's recurrence, does not use its complex state or MIMO in the hot path, and has no LoRA code. Meanwhile, LoRA-for-audio is already public (generation/TTS). The honest position: SSM+LoRA is **internal tooling** that may make a specific effect cheaper or better; the *market* differentiator is the audible result and the finished product, which is what the rest of the text already says. Parts of the text contradict each other on this point; resolve in favor of the product.
2. **Distribution is under-weighted relative to evidence.** The pricing/marketing section treats demos as the bottleneck; the market data says **reach** is (creator review → £2,500 in two days; free capture ecosystems crushing paid-tone prices). A weekly cadence with no borrowed audience or niche channel is a weak wedge.
3. **Timeline vs. conversion.** 90 days is enough to produce artifacts (and this repo is partway there). It is not enough to *close* in a relationship-driven niche; expect two cycles to first paid prototype, and plan the 90 days around artifacts + outreach that continues past day 90.

**Missing from the plan:** JUCE/AAX/Apple costs and gates; the CLAP-only format exposure; a defined contract template (scope, revisions, IP vs licence, data handling); a quality gate that is not a low validation loss (listening tests, null tests, level-matched A/B, and DAW playback on 3+ hosts); and explicit kill criteria.

---

## 10. Concrete edits if you publish the text

1. Replace the Mamba-3 hedge with: *"Mamba-3 is real as of March 2026 (arXiv:2603.15569, ICLR 2026; code in state-spaces/mamba). Cite it precisely — and be aware that most 'Mamba-3' audio implementations, including this one, approximate its recurrence rather than reproduce it."*
2. Add one sentence on JUCE/format reality: *"JUCE targets VST3/AU/AAX natively; CLAP needs the clap-juce-extensions project, and CLAP hosts are still a minority (Bitwig, REAPER, FL Studio)."*
3. Change the CPU line to: *"Measure per-sample cost on the target laptop at 48/96 kHz. A single small SSM layer costs ≈9–13 % of one core; every doubling of depth roughly doubles that. Budget from measurements, not from the paper."*
4. Downgrade LoRA: *"LoRA is an internal efficiency tool (merged adapters, one base model). It is not the product and not a research differentiator; ship merged per-preset weights until many adapters must coexist."*
5. Fix the economics: *"A defined prototype: $2k–8k. A sellable plugin: five figures and months. Owning a plugin usually loses to freelancing until you have distribution — the demo is necessary, not sufficient."*
6. Add a legal paragraph: *"Public music datasets (e.g., MUSDB18) are non-commercial; label lawsuits (RIAA v. Suno/Udio) have produced licensing settlements and nine-figure exposure. Client work needs a data-processing clause."*
7. Add the AI caution with a number: *"AI assistants measured 19 % slower for experienced devs on familiar repos in one RCT (and faster on boilerplate/new code); the failure mode is feeling faster while being slower. Trust the profiler and the listening test."*

---

## Appendix A — benchmark reproduction

CMake is not installed in this sandbox, so I built the project's core and benchmark directly:

```bash
g++ -std=c++20 -O3 -mavx2 -mfma -fno-math-errno -fno-trapping-math \
    -fno-reciprocal-math -fno-signaling-nans -I src \
    -o /tmp/bench tests/benchmark_ssm_block.cpp src/core/ssm_block.cpp \
    src/core/ssm_inference_core.cpp src/core/weight_loader.cpp
/tmp/bench
```

Raw result excerpt (2 vCPU container; treat as a lower bound):

```
  1  48.0 kHz  256   2.664 us/sample   [PASS]
  2  48.0 kHz  256   5.401 us/sample   [PASS]
  6  48.0 kHz  256  22.501 us/sample   [EXCEEDS]
Configurations meeting < 10.0 us/sample budget: 24 / 36 (66.7%)
OVERALL VERIFICATION: PARTIAL — 24/36 configurations pass.
```

## Appendix B — key sources

- Mamba-3 paper: arXiv:2603.15569 (ICLR 2026); code: github.com/state-spaces/mamba (Mamba-3 release 2026-03-17; maintenance commit 2026-07-22)
- Mamba (arXiv:2312.00752); Mamba-2 (arXiv:2405.21060, ICML 2024)
- LoRA: arXiv:2106.09685 (merge semantics; no added inference latency when merged)
- SI-SDR limits: arXiv:2508.14623; SI-SDR loss bound by reference SNR; auraloss (github.com/csteinmetz1/auraloss)
- RT-SEMamba (real-time distilled SSM speech enhancement, 25 ms latency, RTF 0.11)
- RNNoise compute/latency reports; Neural Amp Modeler CPU reports (10–30 %/instance); RTNeural (arXiv:2106.03037)
- RunPod public pricing (2026): RTX 4090 $0.34/hr Community, $0.74/hr Secure; A5000 from $0.16–0.27/hr; network storage $0.07/GB/mo
- JUCE 8 licence tiers (JUCE forum EULA discussion); clap-juce-extensions (free-audio); AAX/PACE preparation writeup (2026); Apple Developer Program $99/yr
- RIAA v. Suno/Udio litigation trackers (2024–2026): Warner settlements (Suno Nov 2025; Udio Oct 2025); UMG/Sony v. Suno ongoing; 61,026 works, $9B+ exposure; MUSDB18 non-commercial licence (Zenodo)
- Freelance/contract and market data: Kush Audio Contract C++/JUCE listing ($50–70/hr, Glassdoor); ZipRecruiter/Glassdoor audio programmer salary bands; custom-plugin build estimates ($30k–100k + 15–25 %/yr); solo-plugin revenue accounts (Nick Thompson blog; JUCE forum thread); The Audio Programmer hiring/community pages
- AI-assistant productivity: METR RCT (2025, 19 % slower; 2026 follow-up ≈18 % faster, CI −38…+9 %); GitHub 55 % task speedup; McKinsey routine-vs-complex split; fp16-CPU slowness reports

*Prices, versions, and litigation status are as of 2026-10-04 and should be re-checked before relying on them.*

---

## Appendix C — Remediation log (2026-10-04)

The 22 verdicts above were the *analysis*. This appendix records what was then changed in
the repository, with the evidence each change rests on. Nothing here upgrades a verdict;
it removes the specific gaps the verdicts identified.

| # | Finding | Change | Evidence it works |
|---|---|---|---|
| 1 | FX plugin constructor segfaulted (misaligned 64-byte `SSMInferenceCore`) | `src/common/aligned_storage.h`; `AlignedStorage<T,64>` used by `MambaSynth.core_storage` and `FxState.core_storage` | `RealtimeSafety.FxStateSaveLoadRoundtrip` 5/5 pass (previously crashed the process) |
| 2 | C++/Python parity test silently skipped (missing vectors) | CI generates `build/parity_*.bin` via `scripts/reference_mamba3.py --generate` before `ctest` | `test_cpp_parity` passes, max abs diff 1.19e-07 |
| 3 | Docs claimed capability the code lacks (exp-trapezoidal recurrence, complex state, active MIMO, conv1d-free latency win, LoRA) | New `ARCHITECTURE_FIDELITY.md` (D1-D6, F1-F5); README / `MIGRATION_TO_MAMBA3.md` / `TRAINING_PLAN.md` / LoRA-companion README corrected | Every claim now carries a `file:line` or a measured number |
| 4 | README headline CPU figure measured a toy shape only | `tests/benchmark_ssm_block.cpp` rewritten with `train4070`/`trainl4`/`max` shapes, `--budget-pct` gate, `--json`; README carries the measured table | 5/24 measured configs meet the 40 %-of-one-core gate; `runpod_l4` is 90-96 % of one core at 1 layer |
| 5 | Trainer had no task-matched objective; weak MSE only | `training/losses.py`: pre-emphasis L1 + SI-SDR + multi-resolution STFT + `hybrid`; wired via `--loss`/config; `mse` retained as default in code, `hybrid` default in configs | `python3 training/losses.py` — 15/15 checks pass; smoke runs on `mse` and `hybrid` both export valid weights |
| 6 | No way to train an effect (dry -> wet) | `PairedAudioDataset` (`dry/` + `wet/` layout, or `--paired`), with an explicit no-delay alignment warning | 3-pair saturation run trains and exports (`sisdr_db` 11.1 at step 1) |
| 7 | Trainer silently exported **untrained** weights when all audio failed to decode | Robust backend chain (`soundfile` -> `torchaudio` -> stdlib `wave`) plus a hard "refusing to export untrained weights" guard | Undecodable-file run now exits 1 with no `.bin` written |
| 8 | CI never ran the gates it advertised | `.github/workflows/ci.yml`: parity vectors, `ctest`, objective self-test, trainer smoke test, real-time gate (40 %), report-only training-shape benchmark + JSON artifact | Workflow reviewed step-by-step; first hosted run pending |
| 9 | State-renormalization artifact (D5) believed to be a parity bug | Reproduced in **both** implementations with a constructed model (|h| = 4): 20.918667 -> 17.822269 at n = 4096, i.e. a 14.8 % gain step every 4096 samples | Same step in C++ and NumPy => design defect, not drift; documented in D5 and `TRAINING_PLAN.md` — **then removed** (D5 closure §0; `scripts/verify_deviations.py` D5 keeps the counterfactual) |

**Two things this appendix does not claim.** (a) The renorm artifact is *characterised*, not
fixed **in this appendix** — it was subsequently removed outright (D5 closure, §0 addendum);
F4 in `ARCHITECTURE_FIDELITY.md` remains the fix path for D1/D2 and needs a listening test,
not just a unit test. (b) The CI workflow has not executed on GitHub yet in this environment
(no hosted runner here) — it is reviewed, not proven; the first push run is the proof.
