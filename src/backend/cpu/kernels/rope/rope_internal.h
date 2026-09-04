#ifndef AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H
#define AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H

/// @file rope_internal.h
/// @brief Internal declarations for the CPU FP32 reference RoPE kernel.
///
/// The prepared args contain only validated, compute-ready POD state. Position
/// values deliberately remain runtime data: a PreparedExecutionBindings may
/// be reused after callers update the position_ids buffer, so validation of
/// its contents belongs in the kernel invocation rather than the params
/// builder.

#include "aethermind/base/status.h"

#include <cstdint>

namespace aethermind::cpu::detail {

/// @brief Pre-validated FP32 arguments for Llama split-half RoPE.
///
/// All pointers and geometry are prepared once for a binding. `position_ids`
/// is read on each execution; its values are validated before the kernel
/// writes either output. `position_divisor` is 1 for standard RoPE and the
/// semantic linear-scaling factor for kLinear.
struct RoPEF32KernelArgs {
    const float* q{};
    const float* k{};
    const int64_t* position_ids{};
    float* q_output{};
    float* k_output{};

    int64_t seq_len{};
    int64_t head_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};

    int64_t q_row_stride{};
    int64_t q_col_stride{1};
    int64_t k_row_stride{};
    int64_t k_col_stride{1};
    int64_t position_stride{1};
    int64_t q_output_row_stride{};
    int64_t q_output_col_stride{1};
    int64_t k_output_row_stride{};
    int64_t k_output_col_stride{1};

    double theta{10000.0};
    double position_divisor{1.0};
};

/// @brief Runs the scalar FP32 Llama split-half RoPE reference kernel.
///
/// @param args Pre-validated tensor layout and frozen RoPE parameters.
/// @return InvalidArgument when a runtime position id is negative; otherwise
///         Ok after rotating both outputs. A position failure occurs before
///         any output write.
Status RunRoPEF32Reference(const RoPEF32KernelArgs& args) noexcept;

} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H
