#include "aethermind/backend/cpu/cpu_info.h"
#include "dot_product_internal.h"

namespace aethermind::cpu {
namespace {

#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
bool HasEffectiveAvx2Fma() noexcept {
    // Cached once per process: detection may issue OS-level requests
    // (e.g. arch_prctl for AMX probing) that are not worth repeating per call.
    static const bool kHasAvx2Fma = [] {
        const auto capabilities = DetectCpuCapabilities();
        return capabilities.ok() &&
               capabilities->effective_features.Contains(CpuFeature::kAvx2) &&
               capabilities->effective_features.Contains(CpuFeature::kFma);
    }();
    return kHasAvx2Fma;
}
#endif

} // namespace

float DotProductF32(const float* a, const float* b, std::size_t n) noexcept {
#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
    if (HasEffectiveAvx2Fma()) {
        return DotProductF32Avx2(a, b, n);
    }
#endif
    return DotProductF32Scalar(a, b, n);
}

void DotProductF32MultiTarget(const float* lhs, const float* rhs, std::size_t n,
                              std::size_t num_targets, std::size_t rhs_stride,
                              float* out) noexcept {
#if defined(DOT_PRODUCT_HAS_AVX2_FMA_KERNEL)
    if (HasEffectiveAvx2Fma()) {
        DotProductF32MultiTargetAvx2(lhs, rhs, n, num_targets, rhs_stride, out);
        return;
    }
#endif
    DotProductF32MultiTargetScalar(lhs, rhs, n, num_targets, rhs_stride, out);
}

} // namespace aethermind::cpu