#ifndef AETHERMIND_CPU_FP16_CONVERSION_INTERNAL_H
#define AETHERMIND_CPU_FP16_CONVERSION_INTERNAL_H

#include <cstddef>

namespace aethermind::cpu {

#if defined(FP16_HAS_F16C_CONVERSION)
void ConvertFp16ToFp32F16c(const std::byte* src, std::byte* dst, size_t count) noexcept;
#endif

#if defined(FP16_HAS_AVX2_CONVERSION)
void ConvertFp32ToFp16Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept;
#endif

} // namespace aethermind::cpu
#endif
