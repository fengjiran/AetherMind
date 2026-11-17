#include "gemm_internal.h"

#if defined(__AVX2__) && defined(__FMA__)
#include "gemm_a32_w16_avx2_internal.h"
#include <bit>
#include <cstring>
#endif

namespace aethermind::cpu::detail {

#if defined(__AVX2__) && defined(__FMA__)
namespace {
struct Bf16GemmWeightOps {
    static __m256 LoadEight(const std::byte* data) noexcept {
        const __m128i bits = _mm_loadu_si128(reinterpret_cast<const __m128i*>(data));
        return _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(bits), 16));
    }
    static float LoadOne(const std::byte* data) noexcept {
        uint16_t bits;
        std::memcpy(&bits, data, sizeof(bits));
        return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16U);
    }
};
} // namespace
#endif

bool TryRunGemmA32Bf16Avx2Fma(const GemmArgs& args) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    return TryGemmA32W16Avx2Fma<Bf16GemmWeightOps>(args);
#else
    return false;
#endif
}

} // namespace aethermind::cpu::detail
