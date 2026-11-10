#ifndef AETHERMIND_CPU_BF16_CONVERSION_INTERNAL_H
#define AETHERMIND_CPU_BF16_CONVERSION_INTERNAL_H

#include <cstddef>

namespace aethermind::cpu {

#if defined(BF16_HAS_AVX2_CONVERSION)
void ConvertBf16ToFp32Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept;
void ConvertFp32ToBf16Avx2(const std::byte* src, std::byte* dst, size_t count) noexcept;
#endif

} // namespace aethermind::cpu
#endif
