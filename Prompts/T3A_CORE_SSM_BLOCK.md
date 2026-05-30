# Worker Task Prompt (T3a) - Core SSM Block Implementation

## ROLE
SSM Inference Engineer implementing the foundational Mamba-SSM block for real-time audio processing.

## CONTEXT
This is the first concrete implementation task: creating a working SSM block that can process audio samples in real-time. This block will form the building block of the larger SSM inference core. We'll start with a simplified version that processes a single channel and focuses on getting the mathematics correct and meeting real-time constraints.

## INSTRUCTIONS
1. Implement a single Mamba-SSM block in C++20 that processes mono audio samples
2. Support the core computation: y = Convolution(x * BC_tilde) * DT_tilde + D * x (simplified form)
3. Implement both per-sample and block processing modes
4. Ensure zero heap allocations and zero locks in the audio processing path
5. Use pre-allocated buffers for state and temporary variables
6. Implement in a way that can be easily extended to multi-channel and multi-layer later
7. Add basic unit tests that verify the mathematical correctness against a reference implementation
8. Benchmark the performance to ensure it meets real-time constraints (<10μs/sample)

## CONSTRAINTS
- **Audio Thread Safety**: The `process_sample()` and `process_block()` methods must contain no heap allocations or mutex locks
- **Memory**: All state (hidden state, etc.) must be pre-allocated in the constructor
- **Precision**: Use FP32 for all computations
- **Interface**: Provide clear methods for initialization, processing, and reset
- **Dependencies**: Only use C++ standard library and possibly Eigen/SIMD intrinsics for optimization
- **Testing**: Must be testable in isolation from CLAP/ARA components
- **Build**: Must work with CMake and compile on at least Linux (extend to macOS/Windows later)

## OUTPUT FORMAT
1. Header file (`ssm_block.h`) containing:
   - Template or concrete class for SSM block
   - Constructor that takes state dimension and pre-allocates buffers
   - Methods: `process_sample(float input)`, `process_block(float* input, float* output, size_t size)`, `reset()`
   - Methods to set SSM parameters (Δ, B, C, D) - these can be called from non-real-time thread
2. Source file (`ssm_block.cpp`) implementing the methods
3. Simple unit test (`test_ssm_block.cpp`) that:
   - Compares against a known-good reference implementation (e.g., Python NumPy)
   - Tests both sample and block processing modes
   - Verifies state reset functionality
4. Basic benchmark (`benchmark_ssm_block.cpp`) that measures:
   - Time per sample for block processing
   - Time per sample for sample processing
   - Memory access patterns

## ACCEPTANCE CRITERIA
- [ ] `process_sample()` and `process_block()` contain no calls to malloc/free/new/delete
- [ ] `process_sample()` and `process_block()` contain no mutex locks
- [ ] Mathematical correctness verified against reference implementation (error < 1e-5)
- [ ] State reset properly clears internal state
- [ ] Performance benchmark shows <10μs/sample for block processing on modern CPU
- [ ] Code compiles with C++20 and passes basic unit tests
- [ ] Clear separation between real-time processing methods and parameter update methods