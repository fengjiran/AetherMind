#include "aethermind/backend/cpu/cpu_bpanel_packing.h"

#include <limits>

namespace aethermind::cpu {

PackingRecipe CpuBPanelF32V1Avx2Recipe() {
    return PackingRecipe{
            .layout = kCpuBPanelF32V1Avx2Layout,
            .alignment = kCpuBPanelF32V1Alignment};
}

StatusOr<size_t> CpuBPanelF32V1PackedByteSize(int64_t n, int64_t k) noexcept {
    if (n < 0 || k < 0) {
        return Status::InvalidArgument(
                "cpu_bpanel_f32 dimensions must be non-negative");
    }

    const auto n_blocks = static_cast<size_t>(
            n / kCpuBPanelF32V1NR + (n % kCpuBPanelF32V1NR != 0));
    const auto k_panels = static_cast<size_t>(
            k / kCpuBPanelF32V1KC + (k % kCpuBPanelF32V1KC != 0));
    size_t elements = n_blocks;
    const size_t factors[] = {
            static_cast<size_t>(kCpuBPanelF32V1KC),
            static_cast<size_t>(kCpuBPanelF32V1NR),
            k_panels,
            sizeof(float),
    };

    for (const size_t factor: factors) {
        if (factor != 0 && elements > std::numeric_limits<size_t>::max() / factor) {
            return Status::Overflow(
                    "cpu_bpanel_f32 packed byte size overflows size_t");
        }
        elements *= factor;
    }
    return elements;
}

} // namespace aethermind::cpu
