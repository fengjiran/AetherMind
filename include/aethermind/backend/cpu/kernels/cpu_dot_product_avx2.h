#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_DOT_PRODUCT_AVX2_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_DOT_PRODUCT_AVX2_H

#include "macros.h"

#include <cstddef>

namespace aethermind {

AM_NODISCARD float DotProductAvx2Unroll(const float* a, const float* b, std::size_t n) noexcept;

}// namespace aethermind

#endif
