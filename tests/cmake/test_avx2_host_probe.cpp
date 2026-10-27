#include "aethermind/backend/cpu/cpu_info.h"

#include <cstdio>
#include <cstdlib>

int main() {
#if !defined(AETHERMIND_AVX2_KERNELS_COMPILED) || !AETHERMIND_AVX2_KERNELS_COMPILED
    std::fputs("AVX2/FMA gate unavailable: candidate kernels were not compiled\n", stderr);
    return EXIT_FAILURE;
#else
    using namespace aethermind;
    const auto capabilities = cpu::DetectCpuCapabilities(CpuFeaturePolicy{
            .required_features = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}),
    });
    if (!capabilities.ok()) {
        std::fprintf(stderr, "AVX2/FMA host gate unavailable: %s\n",
                     capabilities.status().message().c_str());
        return EXIT_FAILURE;
    }
    std::puts("AVX2/FMA host gate passed: compiled candidates and OS-usable hardware verified");
    return EXIT_SUCCESS;
#endif
}
