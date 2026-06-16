#ifndef CONFIG_H
#define CONFIG_H

// === Mamba-3 Architecture Dimensions ===
//
// Mamba-3 eliminates conv1d entirely (0-sample latency), uses multi-head
// SSM with complex-valued state, exponential-trapezoidal discretization,
// MIMO rank-R coupling, RoPE rotary embeddings, and FiLM conditioning.
//
// These MAX_* constants define compile-time buffer sizes. The actual
// model dimensions are read from the weight file header at load time
// and validated against these maxima by WeightLoader::load().

// Core dimensions
#define MAX_D_STATE       32   // N: state dimension (Mamba-3 needs N/2 vs M1 for same quality)
#define MAX_D_MODEL       256  // embedding/input dimension
#define MAX_D_INNER       512  // expand * d_model (expand=2, i.e., 2*d_model)
#define MAX_HEADDIM       64   // head dimension (= d_inner / nheads)
#define MAX_NHEADS         8   // number of heads (= d_inner / headdim)

// MIMO and group dimensions
#define MAX_MIMO_RANK      4   // MIMO rank R (rank-1 = SISO per head; rank>1 = MIMO coupling)
#define MAX_NUM_BC_HEADS   1   // B/C head groups (typically ngroups=1)

// RoPE rotary embedding
#define MAX_NUM_ROPE_ANGLES 16 // d_state/2 (each angle handles 2 state dims)

// Chunk and block sizes
#define MAX_CHUNK_SIZE    128  // for block processing (Mode B)
#define MAX_BLOCK_SIZE    512  // max audio block per process() call
#define MAX_PARAM_ATOMS   256  // SPSC ring buffer capacity

// Scale
#define MAX_LAYERS          6  // maximum model depth
#define COND_DIM          128  // FiLM conditioning dimension (cross-track semantic + f0 + loudness)
#define SSM_STATE_NORM_INTERVAL 4096 // periodic normalization interval

// Versioning
#define VERSION_MAJOR 0
#define VERSION_MINOR 3
#define VERSION_PATCH 0

#endif
