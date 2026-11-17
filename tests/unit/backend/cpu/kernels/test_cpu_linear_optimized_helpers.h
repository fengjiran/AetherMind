#ifndef AETHERMIND_TEST_CPU_LINEAR_OPTIMIZED_HELPERS_H
#define AETHERMIND_TEST_CPU_LINEAR_OPTIMIZED_HELPERS_H

#include "aethermind/backend/cpu/cpu_backend.h"
#include <string>

namespace aethermind::test {

// Expected routing policy, independent of registry contents and kernel output.
inline std::string ExpectedA32ProjectionKernelName(
        OpType op, DataType weights, WeightFormat format = WeightFormat::kPlain,
        CpuFeatureSet features = CpuBackend{}.cpu_capabilities().effective_features) {
    bool vector = false;
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    vector = weights != DataType::Float(16) &&
             features.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}));
#endif
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
    vector = vector || (weights == DataType::Float(16) &&
                        features.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c})));
#endif
    std::string name = op == OpType::kLinear ? "cpu::linear_f32_" : op == OpType::kQkvLinear ? "cpu::qkv_linear_f32_"
                                                                                             : "cpu::gate_up_linear_f32_";
    name += weights == DataType::Float32() ? "w32" : weights == DataType::Float(16) ? "wf16"
                                                                                    : "wbf16";
    if (op == OpType::kLinear && format == WeightFormat::kPacked) {
        name += "_packed_identity";
    }
    name += vector ? "_avx2_fma" : weights == DataType::Float32() ? "_scalar"
                                                                  : "_reference";
    return name;
}

} // namespace aethermind::test
#endif
