# Integration Contract Prompt (T1.5) - Memory Schemas & Thread Boundaries

## ROLE
Integration Specialist defining the critical contracts between audio thread and non-real-time systems.

## CONTEXT
The audio thread operates under severe constraints: zero heap allocations, zero mutex locks, and deterministic timing. All memory management must be pre-allocated, and all synchronization must be lock-free. This prompt defines exactly how memory is owned, shared, and accessed between the real-time audio processing domain and the control/UI/ML domains.

## INSTRUCTIONS
1. Define precise memory ownership rules for all data crossing the audio/non-realtime boundary
2. Specify lock-free communication mechanisms (ring buffers, atomic variables, etc.)
3. Detail the exact memory layout and alignment requirements for audio buffers
4. Define thread boundaries: what runs on audio thread vs. worker threads vs. GUI thread
5. Specify synchronization primitives allowed (atomics, memory fences) and forbidden (mutexes, condition variables)
6. Define the lifecycle of plugin instances, memory allocation/deallocation points
7. Establish buffer size policies and alignment requirements for SIMD processing

## CONSTRAINTS
- **Audio Thread**: Must never call malloc/free, new/delete, or any blocking OS APIs
- **Memory**: All audio buffers must be pre-allocated during initialization
- **Alignment**: Audio buffers must be aligned to 64-byte cache lines for optimal SIMD
- **Communication**: Only lock-free mechanisms permitted between threads
- **Atomics**: Only std::atomic with appropriate memory ordering allowed
- **Preemption**: Audio thread must not be blocked by any other thread
- **Priority Inversion**: Must be avoided through proper priority inheritance or wait-free design

## OUTPUT FORMAT
Detailed specification including:
1. Memory ownership matrix showing who allocates/owns/frees each data structure
2. Thread assignment table for all major subsystems
3. Lock-free queue specifications (capacity, element type, ordering guarantees)
4. Atomic variable usage guidelines with memory ordering requirements
5. Buffer alignment and size requirements
6. Initialization/cleanup sequence and where allocations are permitted
7. Error handling strategy that doesn't violate real-time constraints

## ACCEPTANCE CRITERIA
- [ ] Clear definition of which thread owns each memory buffer and data structure
- [ ] Specification of all lock-free communication mechanisms with performance characteristics
- [ ] Memory alignment requirements for SSE/AVX/AVX2/AVX-512 compatibility
- [ ] Prohibited operations list for audio thread (syscalls, library calls, etc.)
- [ ] Initialization sequence showing where memory allocation is safely permitted
- [ ] Cleanup procedure ensuring no memory leaks or use-after-free
- [ ] Diagram showing data flow between threads with synchronization points