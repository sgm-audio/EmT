# Mamba-SSM CLAP/ARA Plugin Suite Prompts

This directory contains a complete hierarchy of system prompts designed to guide an AI coding team in building a suite of CLAP 1.1+ and ARA 2.0 plugins powered by Mamba-SSM state space models.

## Directory Structure

```
Prompts/
├── MASTER_SYSTEM_PROMPT.md              # T1 - System Architect
├── INTEGRATION_CONTRACT_PROMPT.md       # T1.5 - Memory Schemas & Thread Boundaries
├── T2A_SSM_INFERENCE_CORE.md            # T2a - SSM C++ Inference Core
├── T2B_CLAP_ARA_HOST_INTEGRATION.md     # T2b - CLAP/ARA Host Integration
├── T2C_ML_EXPORT_WEIGHT_PIPELINE.md     # T2c - ML Export & Weight Pipeline
├── T2D_UI_PARAMETER_STATE_MANAGEMENT.md # T2d - UI/Parameter/State Management
├── T3A_CORE_SSM_BLOCK.md                # T3a - Core SSM Block Implementation
├── T3B_CLAP_NOTE_PORTS.md               # T3b - CLAP Note Ports Implementation
├── T3C_ML_EXPORT_PIPELINE.md            # T3c - ML Export Pipeline Implementation
└── README.md                            # This guide
```

## Plugin Suite Overview

The prompts guide the creation of three interconnected plugins:

1. **Mamba Synth** - Generative audio plugin conditioned on MIDI/ARA
2. **Mamba FX** - Long-context neural audio processing effect
3. **ARA Generative Companion** - Offline region generation via ARA renderers

## Technical Requirements Addressed

- Zero-allocation, zero-lock audio thread semantics
- C++20, cross-platform (macOS/Win/Linux), CMake build
- Per-sample and block-based Mamba-SSM inference modes
- Full CLAP 1.1+ extensions: audio-ports, note-ports, params, gui, thread-check, state, tail
- Complete ARA 2.0 implementation: DocumentController, PlaybackRenderer, EditorRenderer
- ML pipeline: PyTorch → quantized flat-binary weights → C++ runtime
- ARA musical context conditioning of SSM parameters (Δ, B, C) via learned projections

## How to Use These Prompts

### For System Architects/Tech Leads
1. Start with `MASTER_SYSTEM_PROMPT.md` to understand the overall vision
2. Review `INTEGRATION_CONTRACT_PROMPT.md` for critical real-time constraints
3. Use the T2 domain prompts to assign work to specialized teams

### For Implementation Teams
- **SSM Core Team**: Begin with `T2A_SSM_INFERENCE_CORE.md` and implement `T3A_CORE_SSM_BLOCK.md`
- **Integration Team**: Start with `T2B_CLAP_ARA_HOST_INTEGRATION.md` and implement `T3B_CLAP_NOTE_PORTS.md`
- **ML Pipeline Team**: Follow `T2C_ML_EXPORT_WEIGHT_PIPELINE.md` and implement `T3C_ML_EXPORT_PIPELINE.md`
- **UI/State Team**: Work from `T2D_UI_PARAMETER_STATE_MANAGEMENT.md`

### Sprint Planning Guidance
Each T3 prompt represents a concrete sprint task:
- T3A: Foundational SSM block (1-2 weeks)
- T3B: CLAP note-ports integration (1 week)
- T3C: Initial ML export pipeline (1-2 weeks)

## Dependencies Between Prompts

```
MASTER_SYSTEM_PROMPT → INTEGRATION_CONTRACT_PROMPT
                              ↓
        ┌───────────────────┼───────────────────┐
        ↓                   ↓                   ↓
T2A_SSM_INFERENCE_CORE  T2B_CLAP_ARA_HOST_INTEGRATION  T2C_ML_EXPORT_WEIGHT_PIPELINE  T2D_UI_PARAMETER_STATE_MANAGEMENT
        ↓                   ↓                   ↓                   ↓
    T3A_CORE_SSM_BLOCK   T3B_CLAP_NOTE_PORTS   T3C_ML_EXPORT_PIPELINE  (UI tasks based on T2D)
```

## Next Steps After These Prompts

Once the foundation is established with these prompts, the team should:
1. Implement the core SSM inference capabilities
2. Build the CLAP/ARA integration layers
3. Establish the ML export pipeline
4. Develop the UI/parameter systems
5. Expand to full multi-layer SSM models
6. Implement advanced conditioning systems
7. Create comprehensive test suites
8. Prepare for distribution and hosting validation

## Validation Checklist

Before considering the suite complete, verify:
- [ ] All audio thread functions are allocation-free and lock-free
- [ ] CLAP validator passes for all required extensions
- [ ] ARA 2.0 integration works with at least one major DAW
- [ ] ML export pipeline produces bit-equivalent results (within quantization error)
- [ ] UI remains responsive under heavy audio load
- [ ] Preset saving/loading works correctly in CLAP and ARA hosts
- [ ] Performance meets real-time requirements (<10μs/sample)