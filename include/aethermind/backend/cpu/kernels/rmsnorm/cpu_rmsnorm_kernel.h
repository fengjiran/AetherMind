#ifndef AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_CPU_RMSNORM_KERNEL_H

/// @file cpu_rmsnorm_kernel.h
/// @brief CPU RMSNorm kernel SDK launch interface.
///
/// Exposes a type-erased entry point for callers that do not depend on
/// TensorView or other engine internals. The backend-internal params
/// (RmsNormKernelParams) and validated args (RmsNormFp32KernelArgs) live in
/// src/backend/cpu/kernels/rmsnorm/rmsnorm_internal.h and are never
/// included by operator code.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>

namespace aethermind {

/// @brief Type-erased arguments for the RMSNorm operation.
///
/// All data pointers are borrowed and must remain valid for the duration of
/// the call. The caller must ensure the backing memory is correctly typed
/// according to @p dtype and is large enough for the addressed elements
/// implied by @p seq_len, @p hidden_size, and the row/column strides.
/// Strides support non-contiguous slices; the default column stride of 1
/// indicates innermost dimension is physically contiguous.
struct RmsNormArgs {
    const void* input = nullptr;
    const void* weight = nullptr;
    void* output = nullptr;
    int64_t seq_len = 0;
    int64_t hidden_size = 0;
    int64_t input_row_stride = 0;
    int64_t input_col_stride = 1;
    int64_t weight_stride = 1;
    int64_t output_row_stride = 0;
    int64_t output_col_stride = 1;
    float eps = 1.0e-5f;
    DataType dtype;
    void* workspace = nullptr;
    size_t workspace_size = 0;
};

/// @brief Launches the CPU RMSNorm kernel with type-erased arguments.
///
/// Validates pointers, shapes, strides, and epsilon before dispatching to
/// the scalar or AVX2 micro-kernel. The AVX2 path is selected only when the
/// column strides are all 1 and the CPU reports AVX2 support via
/// GetCpuFeatures(); otherwise the scalar path is used.
///
/// @param args Type-erased RMSNorm arguments. Pointers must be non-null,
///             @p seq_len must be non-negative, @p hidden_size must be
///             positive, @p eps must be finite and positive, and
///             @p workspace / @p workspace_size must be nullptr / 0 in v1.
/// @return OkStatus on success, InvalidArgument when preconditions are
///         violated, or Unimplemented for unsupported @p dtype.
/// @note This is an SDK thin wrapper and does not go through KernelRegistry;
///       the performance path uses ExecutionPlan with CpuBackend::PrepareKernel.
///       The function is noexcept and reports errors only via the return value.
Status LaunchRmsNorm(const RmsNormArgs& args) noexcept;

}// namespace aethermind

#endif
