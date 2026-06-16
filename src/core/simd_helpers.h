#ifndef SIMD_HELPERS_H
#define SIMD_HELPERS_H

#include <cstddef>
#include <cstdint>

#if defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
#include <emmintrin.h>
#define MAMBA_SSE2 1
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define MAMBA_NEON 1
#endif

namespace mamba {
namespace simd {

// Dot product: sum(a[i] * b[i]) for i in [0, n).
// n must be a multiple of 4 when SSE2 is available.
inline float dot_product(const float* __restrict a, const float* __restrict b, size_t n) {
    float sum = 0.0f;
#if MAMBA_SSE2
    __m128 acc = _mm_setzero_ps();
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        acc = _mm_add_ps(acc, _mm_mul_ps(va, vb));
    }
    float tmp[4];
    _mm_storeu_ps(tmp, acc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += a[i] * b[i];
#elif MAMBA_NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vmlaq_f32(acc, va, vb);
    }
    float tmp[4];
    vst1q_f32(tmp, acc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];
    for (; i < n; ++i) sum += a[i] * b[i];
#else
    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
#endif
    return sum;
}

// Element-wise multiply-add: out[i] += a[i] * scalar, for i in [0, n).
inline void fmadd_scalar(float* __restrict out, const float* __restrict a, float scalar, size_t n) {
#if MAMBA_SSE2
    __m128 vs = _mm_set1_ps(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vo = _mm_loadu_ps(out + i);
        vo = _mm_add_ps(vo, _mm_mul_ps(va, vs));
        _mm_storeu_ps(out + i, vo);
    }
    for (; i < n; ++i) out[i] += a[i] * scalar;
#elif MAMBA_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vo = vld1q_f32(out + i);
        vo = vmlaq_f32(vo, va, vs);
        vst1q_f32(out + i, vo);
    }
    for (; i < n; ++i) out[i] += a[i] * scalar;
#else
    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; ++i) out[i] += a[i] * scalar;
#endif
}

// Element-wise multiply: out[i] = a[i] * b[i], for i in [0, n).
inline void mul_elementwise(float* __restrict out, const float* __restrict a, const float* __restrict b, size_t n) {
#if MAMBA_SSE2
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
#elif MAMBA_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];
#else
    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

// Element-wise multiply-add: out[i] = a[i] * b[i] + c[i], for i in [0, n).
inline void fmadd_elementwise(float* __restrict out, const float* __restrict a, const float* __restrict b, const float* __restrict c, size_t n) {
#if MAMBA_SSE2
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 vc = _mm_loadu_ps(c + i);
        _mm_storeu_ps(out + i, _mm_add_ps(_mm_mul_ps(va, vb), vc));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#elif MAMBA_NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        vst1q_f32(out + i, vmlaq_f32(vc, va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#else
    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#endif
}

// Element-wise scale: out[i] = a[i] * scalar, for i in [0, n).
inline void scale(float* __restrict out, const float* __restrict a, float scalar, size_t n) {
#if MAMBA_SSE2
    __m128 vs = _mm_set1_ps(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        _mm_storeu_ps(out + i, _mm_mul_ps(va, vs));
    }
    for (; i < n; ++i) out[i] = a[i] * scalar;
#elif MAMBA_NEON
    float32x4_t vs = vdupq_n_f32(scalar);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(out + i, vmulq_f32(va, vs));
    }
    for (; i < n; ++i) out[i] = a[i] * scalar;
#else
    #pragma clang loop vectorize(enable)
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * scalar;
#endif
}

} // namespace simd
} // namespace mamba

#endif