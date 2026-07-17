/// Internal declarations for the CPU Add kernel (directory-structured).
///
/// Declares the private scalar execution entry point. The public `CpuAddParams`
/// type and `CpuAddKernel` declaration are in the public header. Validation and
/// kernel registration live in `add_entry.cpp`; the scalar implementation lives
/// in `add_scalar.cpp`.

#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H

#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <array>
#include <cstdint>

namespace aethermind::cpu::detail {

constexpr uint32_t kMaxRank = ShapeAndStride::kMaxRank;

/// Pre-validated, type-erased kernel arguments for the Add implementation.
///
/// Produced by the entry layer (ValidateAndBuildArgs) and consumed by the
/// implementation layer (AddKernel_Scalar). Separates the "what to validate"
/// concern from the "how to compute" concern.
struct AddKernelArgs {
    const void* lhs_data = nullptr;
    const void* rhs_data = nullptr;
    void* output_data = nullptr;
    DataType dtype;
    int64_t numel = 0;
    bool is_flat = false;

    // Broadcast / strided path fields (only used when !is_flat).
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

/// Executes element-wise add via scalar loops for all supported dtypes.
///
/// Selects between a flat loop (when is_flat is true) and a stride-aware
/// scalar loop (for general broadcasts including non-contiguous output).
/// Templates are TU-local in add_scalar_impl.h, instantiated in add_scalar.cpp.
Status AddKernel_Scalar(const AddKernelArgs& args) noexcept;

}// namespace aethermind::cpu::detail

#endif// AETHERMIND_BACKEND_CPU_KERNELS_ADD_ADD_INTERNAL_H