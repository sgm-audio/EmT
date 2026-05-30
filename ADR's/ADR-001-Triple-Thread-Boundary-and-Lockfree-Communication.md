# ADR-001: Triple-Thread Boundary and Lock-Free Communication

## Status
Accepted

## Context
We need to define the threading model for the plugin suite to ensure real-time safety in the audio thread while allowing communication from the main thread (UI, state serialization) and the ARA thread (offline rendering, analysis). The audio thread must be free of allocations, locks, and blocking operations.

## Decision
We define three threads:
1. **Audio Thread**: Runs the audio callback (`process()`). It is real-time and must never block.
   - Owns the current SSM state (`h_state`).
   - Reads parameters via atomic float array or lock-free ring buffer (for smoothed values).
   - Does not own any memory that is allocated/freed by other threads during real-time processing.

2. **Main Thread**: Handles UI events, CLAP parameter flushes, state serialization, and weight loading.
   - Responsible for heavy computations (e.g., weight loading, state serialization).
   - Communicates parameter changes to the audio thread via lock-free structures.
   - Owns the GUI and is responsible for updating it.

3. **ARA Thread**: Used by the host for ARA-specific tasks (DocumentController updates, offline rendering, analysis).
   - Runs ARA callbacks that may be time-consuming but are not part of the real-time audio chain.
   - Updates the conditioning frame that influences the SSM parameters (Δ, B) via learned projections.
   - Communicates with the audio thread via a double-buffered, atomic pointer to a conditioning frame.

### Lock-Free Communication Mechanisms
- **Parameter Changes**: Use an atomic float array (for simplicity and cache efficiency) or a single-producer-single-consumer (SPSC) ring buffer (e.g., `moodycamel::ReaderWriterQueue`) for transmitting parameter updates from the main thread to the audio thread. The audio thread reads the latest value without blocking.
- **ARA Conditioning**: Use a double-buffered structure for the conditioning frame (`ConditioningFrame`). The ARA thread writes to the back buffer and then atomically swaps a pointer to make it the front buffer. The audio thread reads the front buffer (via an atomic pointer) without blocking. The structure is aligned to 128 bits (16 bytes) to avoid false sharing.
- **State Snapshots**: For state serialization (e.g., when saving a preset), the audio thread writes the current SSM state (`h_state`) to a secondary buffer (pre-allocated) and then increments a sequence lock (seqlock). The main thread, when serializing state, reads the seqlock, copies the secondary buffer, and then reads the seqlock again to ensure consistency. If the seqlock changed during the copy, it retries.

## Consequences
- The audio thread is guaranteed to be free of locks, allocations, and blocking I/O.
- Parameter updates are eventually consistent and may be delayed by up to one audio buffer period, but are smooth and glitch-free.
- ARA conditioning updates are also eventually consistent and safe for real-time reading.
- State snapshots are consistent without blocking the audio thread for long periods.

## Implementation Notes
- The atomic float array for parameters must be aligned to avoid tearing (use `std::atomic<float>` for each parameter or align the whole array to cache line size).
- The double-buffered conditioning frame must be padded to avoid false sharing.
- The seqlock for state snapshots can be implemented with a sequence counter (even when writing, odd when reading) or a simple atomic integer with a retry loop.

## References
- [Lock-Free Programming](https://www.justsoftwaresolutions.co.uk/threading/lockfree_programming.html)
- [Sequential Locks (Seqlock)](https://en.wikipedia.org/wiki/Seqlock)
- [moodycamel::ReaderWriterQueue](https://github.com/cameron314/concurrentqueue)