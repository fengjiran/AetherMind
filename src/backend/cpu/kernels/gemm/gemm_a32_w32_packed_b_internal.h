#ifndef AETHERMIND_BACKEND_CPU_KERNELS_GEMM_A32_W32_PACKED_B_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_GEMM_A32_W32_PACKED_B_INTERNAL_H

#include "aethermind/base/status.h"

namespace aethermind::cpu::detail {

struct PackedGemmF32Args;

/// @brief Validates dimensions, artifact metadata and alignment for one call.
///
/// ISA-neutral: shared by the checked SIMD entry and the scalar reference
/// entry so both accept exactly the same requests.
Status ValidatePackedBArgs(const PackedGemmF32Args& packed) noexcept;

/// @brief Executes the packed-B reference for a validated, nonempty output.
/// Requires valid artifact metadata and C storage. A and B are read for K > 0.
void RunPackedBReferenceValidated(const PackedGemmF32Args& packed) noexcept;

} // namespace aethermind::cpu::detail

#endif
