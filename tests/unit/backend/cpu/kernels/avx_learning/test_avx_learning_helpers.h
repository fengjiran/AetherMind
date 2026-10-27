#ifndef AETHERMIND_TEST_AVX_LEARNING_HELPERS_H
#define AETHERMIND_TEST_AVX_LEARNING_HELPERS_H

#include "aethermind/base/macros.h"

#include <cstddef>
#include <cstdint>

namespace aethermind::test::avx_learning {

// Call only after the baseline test wrapper verifies usable AVX2 and FMA.
// Fixed-width operations require eight input/output elements unless stated.
// No SIMD types cross this boundary, and no helper is inlined into a caller.
AM_NOINLINE void Add(const float* a, const float* b, float* out, std::size_t n) noexcept;
AM_NOINLINE void FusedMultiplyAdd(const float* a, const float* b, const float* c,
                                  float* out, std::size_t n) noexcept;
AM_NOINLINE void Broadcast8(float value, float* out) noexcept;
AM_NOINLINE float HorizontalSum8(const float* data) noexcept;
AM_NOINLINE float HorizontalMax8(const float* data) noexcept;
AM_NOINLINE void ShuffleWithinLane8(const float* data, float* out) noexcept;
AM_NOINLINE void Permute8(const float* data, const int32_t* indices, float* out) noexcept;
AM_NOINLINE void ClampNegativeToZero8(const float* data, float* out) noexcept;
AM_NOINLINE float SumSquares8(const float* data) noexcept;
AM_NOINLINE void ScaleRmsNorm8(const float* row, const float* weight, float inv_rms,
                               float* out) noexcept;
AM_NOINLINE void Gather8(const float* table, const int32_t* indices, float* out) noexcept;
AM_NOINLINE void SquareWithScalarTail(const float* data, float* out, std::size_t n) noexcept;
// Reads two elements and writes four elements, zeroing the unused two lanes.
AM_NOINLINE void MaskedLoadTwo(const float* data, float* out) noexcept;

} // namespace aethermind::test::avx_learning

#endif
