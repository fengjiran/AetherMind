#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "utils/overflow_check.h"

namespace aethermind::cpu {

StatusOr<size_t> CpuBPanelF32Kc512Nr16PackedByteSize(int64_t n, int64_t k) noexcept {
    if (n < 0 || k < 0) {
        return Status::InvalidArgument(
                "cpu_bpanel_f32 dimensions must be non-negative");
    }

    const auto n_blocks = static_cast<size_t>(
            n / kCpuBPanelF32Kc512Nr16NR + (n % kCpuBPanelF32Kc512Nr16NR != 0));
    const auto k_panels = static_cast<size_t>(
            k / kCpuBPanelF32Kc512Nr16KC + (k % kCpuBPanelF32Kc512Nr16KC != 0));
    size_t elements = n_blocks;
    const size_t factors[] = {
            static_cast<size_t>(kCpuBPanelF32Kc512Nr16KC),
            static_cast<size_t>(kCpuBPanelF32Kc512Nr16NR),
            k_panels,
            sizeof(float),
    };

    for (const size_t factor: factors) {
        if (CheckOverflowMul(elements, factor, &elements)) {
            return Status::Overflow(
                    "cpu_bpanel_f32 packed byte size overflows size_t");
        }
    }
    return elements;
}

} // namespace aethermind::cpu
