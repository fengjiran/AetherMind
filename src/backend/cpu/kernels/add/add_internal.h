#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H

/// @file add_internal.h
/// @brief Backend-internal Add compute-ready params and reference micro-kernel.
///
/// Defines the pre-validated `AddKernelArgs` and the dtype-generic reference Add
/// micro-kernel. The kernel entry itself is TU-local to `add_entry.cpp`;
/// operators never include this header.

#include "aethermind/backend/cpu/kernels/common/broadcast_utils.h"
#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <array>

namespace aethermind::cpu::detail {

/// @brief Pre-validated, type-erased arguments for Add micro-kernels.
///
/// Produced by `ValidateAndBuildArgs` (the dtype-agnostic Add shell over the
/// shared `ValidateAndBuildElementwiseArgs` template) from the binding-time
/// `KernelParamsBuildContext` and consumed by the reference implementation,
/// separating validation from compute. `numel` is the broadcast output element
/// count; a zero count means the entry returns before dispatch.
struct AddKernelArgs {
    const void* lhs_data = nullptr;
    const void* rhs_data = nullptr;
    void* output_data = nullptr;
    DataType dtype;
    int64_t numel = 0;
    bool is_flat = false;

    // Broadcast path metadata, read only when !is_flat. Coordinates are
    // decomposed over output_rank and bounded by kMaxRank.
    int32_t lhs_rank = 0;
    int32_t rhs_rank = 0;
    int32_t output_rank = 0;
    std::array<int64_t, kMaxRank> lhs_shape{};
    std::array<int64_t, kMaxRank> lhs_strides{};
    std::array<int64_t, kMaxRank> rhs_shape{};
    std::array<int64_t, kMaxRank> rhs_strides{};
    std::array<int64_t, kMaxRank> output_shape{};
    std::array<int64_t, kMaxRank> output_strides{};
};

/// @brief Runs the reference Add micro-kernel for every supported dtype.
///
/// Dispatches on `args.dtype`, then selects a flat loop when `args.is_flat`
/// and a stride-aware broadcast loop otherwise.
///
/// @param args Pre-validated kernel arguments. Data pointers must be
///        non-null, `numel` positive, and `dtype` one of
///        `kAddSupportedDTypes`.
/// @return Ok on success, `kOverflow` when integer addition overflows, or
///         `kInvalidArgument` for an unsupported dtype.
Status RunAddReference(const AddKernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
