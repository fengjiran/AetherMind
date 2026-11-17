#include "aethermind/dtypes/half.h"
#include "gemm_internal.h"

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
#include "gemm_a32_w16_avx2_internal.h"
#include <cstring>
#endif

namespace aethermind::cpu::detail {

#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
namespace {
struct F16GemmWeightOps {
    static __m256 LoadEight(const std::byte* data) noexcept {
        return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data)));
    }
    static float LoadOne(const std::byte* data) noexcept {
        uint16_t bits;
        std::memcpy(&bits, data, sizeof(bits));
        return Half(bits, Half::from_bits());
    }
};
} // namespace
#endif

bool TryRunGemmA32F16Avx2Fma(const GemmArgs& args) noexcept {
#if defined(__AVX2__) && defined(__FMA__) && defined(__F16C__)
    return TryGemmA32W16Avx2Fma<F16GemmWeightOps>(args);
#else
    return false;
#endif
}

} // namespace aethermind::cpu::detail
