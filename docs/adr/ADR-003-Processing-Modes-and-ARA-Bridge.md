# ADR-003: Processing Modes and ARA Bridge

## Status
Accepted

## Context
We need to support two inference modes for the Mamba-SSM model to accommodate different model sizes and latency requirements.
Additionally, we need to define how ARA musical context (tempo, key, chords) is used to condition the SSM parameters (Δ and B) via learned projections.

## Decision
### Processing Modes
We define two modes:

**Mode A (Tiny)**: 
- Designed for very small Mamba models (e.g., ≤2M parameters) where per-sample or micro-block processing is feasible.
- Processes audio in blocks of 4 to 16 samples (or even per-sample) to minimize latency.
- The state is updated per sample or per micro-block. There is no significant state carryover between blocks that would require overlap-add.
- Suitable for applications where latency is critical (e.g., live performance).

**Mode B (Block)**:
- Designed for larger models (e.g., >2M parameters) where processing per sample would be too expensive.
- Processes audio in blocks of 64 to 512 samples.
- The state is carried over from one block to the next. The SSM is inherently stateful, so we simply feed the final state of one block as the initial state of the next.
- If the model is non-causal (which typical Mamba-SSM for audio is not, but we consider the possibility), we would use overlap-add. However, for causal models, overlap-add is not necessary.
- We choose the block size based on the model size and the available CPU budget to stay within the 30% real-time budget.

### ARA Bridge
The ARA thread provides musical context (tempo, key, chord progression) at a rate much lower than the audio sample rate (e.g., every 100ms or on change).
We convert this musical context into a conditioning vector `cond` of fixed dimension `COND_DIM` (e.g., 16 or 32) via a predefined mapping (which may be learned or heuristic).
This conditioning vector is then used to bias the SSM parameters Δ and B via learned projections:
- For each layer, we have a small neural network (e.g., a single linear layer) that maps `cond` to a bias for Δ and another for B.
- The biases are added to the base Δ and B (which are stored in the weights).
- The projections are precomputed and stored as part of the weight file (or computed on the fly from the weights, but we prefer to precompute for efficiency).
- The conditioning vector is updated by the ARA thread and shared with the audio thread via the double-buffered mechanism described in ADR-001.

## Consequences
- The system can handle a wide range of model sizes by switching between Mode A and Mode B (and adjusting block size) without changing the core inference code.
- The ARA conditioning is applied in a way that is efficient and real-time safe: the audio thread reads the latest conditioning vector (via atomic pointer) and applies the biases via a few multiply-add operations (if the projections are linear).
- The design isolates the real-time audio thread from the variability of ARA updates.

## Implementation Notes
- We define two processing functions in the SSM core: `process_tiny` (for Mode A) and `process_block` (for Mode B). The plugin selects which one to use based on the model size and user preference (or automatically).
- The conditioning vector is stored in a struct that is aligned to 128 bits (16 bytes) to avoid false sharing and to allow atomic pointer swaps (if we use a double buffer and an atomic pointer to the struct).
- The projections for Δ and B are applied as: 
    Δ_effective = Δ_base + W_delta * cond
    B_effective = B_base + W_bias * cond
  where W_delta and W_bias are matrices (or vectors, if we output a scalar bias per layer) that are part of the weight file.
- We precompute the biases for the current conditioning vector in the ARA thread (if the projection is expensive) or in the audio thread (if it's cheap, like a single linear layer). Given that the conditioning vector changes infrequently, we can compute the biases in the ARA thread and then share the effective Δ and B (or the biases) with the audio thread. However, to keep the audio thread simple, we share the conditioning vector and let the audio thread compute the biases (if the projection is lightweight) or we precompute the biases in the ARA thread and share them. We choose the latter: the ARA thread computes the biases for Δ and B for each layer and stores them in a buffer that is double-buffered and shared with the audio thread. This way, the audio thread only needs to add the biases to the base parameters.

## References
- [Mamba-SSM paper](https://arxiv.org/abs/2307.09323)
- [Real-time audio processing](https://ccrma.stanford.edu/~stilti/papers/jamasp.pdf)