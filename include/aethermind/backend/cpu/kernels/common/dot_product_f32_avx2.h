#ifndef AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_F32_AVX2_H
#define AETHERMIND_BACKEND_CPU_KERNELS_COMMON_DOT_PRODUCT_F32_AVX2_H

#include "aethermind/base/macros.h"

#include <cstddef>

namespace aethermind {

AM_NODISCARD float DotProductF32Avx2Unroll(const float* a, const float* b, std::size_t n) noexcept;

} // namespace aethermind

#endif
