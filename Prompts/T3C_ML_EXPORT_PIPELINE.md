# Worker Task Prompt (T3c) - ML Export Pipeline Implementation

## ROLE
ML Infrastructure Engineer implementing the initial PyTorch to flat-binary weight export pipeline for Mamba-SSM models.

## CONTEXT
This task establishes the ML export pipeline that will convert trained Mamba-SSM models from PyTorch to the flat-binary format required by the C++ inference core. We'll start with a simple Mamba block implementation in PyTorch, implement the export script, define the binary format, and create a basic C++ loader. This creates the essential bridge between training and inference.

## INSTRUCTIONS
1. Define the flat-binary weight file format specification for a single Mamba-SSM block
2. Implement a simple Mamba-SSM block in PyTorch for testing
3. Create a PyTorch export script that:
   - Takes the trained PyTorch Mamba block
   - Extracts weights in deterministic order
   - Applies INT8 quantization (or FP16) to weights
   - Packs weights and metadata into the flat-binary format
4. Implement a C++ weight loader that:
   - Reads and validates the flat-binary format
   - Extracts metadata
   - Dequantizes weights
   - Provides access to weights for the SSM inference core
5. Create a validation script that compares PyTorch and C++ inference outputs
6. Ensure all components are deterministic and reproducible

## CONSTRAINTS
- **Determinism**: Same PyTorch model → identical weight binary every time
- **Format**: Little-endian, naturally aligned data structures
- **Quantization**: Choose INT8 or FP16 with <1% error for audio weights
- **Metadata**: Must include state dimension, any necessary configuration
- **Dependencies**: 
  - PyTorch export: torch, numpy
  - C++ loader: only C++ standard library
  - Validation: PyTorch, numpy, C++ implementation
- **Testing**: Must be testable in isolation from CLAP/ARA components
- **Size**: Binary should be reasonably sized (quantization helps)

## OUTPUT FORMAT
1. Format specification document (`WEIGHT_FORMAT_SPEC.md`) detailing:
   - Magic number and version
   - Metadata structure (state dim, etc.)
   - Weight layout and quantization scheme
2. PyTorch export script (`export_mamba_block.py`) with:
   - Model loading/weight extraction
   - Quantization implementation
   - Binary writing with metadata
3. C++ weight loader class (`MambaWeightLoader`) with:
   - File/memory buffer loading
   - Magic number and version validation
   - Metadata extraction
   - Weight access methods (dequantized)
4. Validation script (`validate_export.py`) that:
   - Loads same weights in PyTorch and C++
   - Runs identical inputs through both
   - Compares outputs within quantization error tolerance
5. Simple test Mamba block in PyTorch for development/testing

## ACCEPTANCE CRITERIA
- [ ] Weight format specification is complete and deterministic
- [ ] Export script produces identical binary given same PyTorch model and settings
- [ ] C++ loader correctly reads binary and extracts metadata
- [ ] Quantization error is <1% for weights (verified by validation script)
- [ ] Validation script shows PyTorch and C++ inference outputs match within quantization error
- [ ] Format includes all necessary metadata for SSM core initialization
- [ ] C++ loader has no external dependencies beyond standard library
- [ ] Export pipeline works for the test Mamba block implementation