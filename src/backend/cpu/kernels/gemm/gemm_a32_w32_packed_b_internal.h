#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_A32_W32_PACKED_B_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_A32_W32_PACKED_B_INTERNAL_H

namespace aethermind::cpu::detail {

struct PackedGemmF32Args;

/// @brief Executes the packed-B reference for a validated, nonempty output.
/// Requires valid artifact metadata and C storage. A and B are read for K > 0.
void RunPackedBReferenceValidated(const PackedGemmF32Args& packed) noexcept;

} // namespace aethermind::cpu::detail

#endif
