#include "gemm_internal.h"

#if defined(__AVX2__) && defined(__FMA__)
#include "gemm_a32_w16_avx2_internal.h"
#endif

namespace aethermind::cpu::detail {

bool TryRunGemmA32WBf16Avx2Fma(const GemmArgs& args) noexcept {
#if defined(__AVX2__) && defined(__FMA__)
    return TryGemmA32W16Avx2Fma<Bf16GemmWeightOps>(args);
#else
    return false;
#endif
}

} // namespace aethermind::cpu::detail
