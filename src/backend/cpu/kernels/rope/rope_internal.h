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
#include "aethermind/operators/op_params.h"
#include "utils/logging.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace aethermind::cpu::detail {

/// Binary attrs prefix. Resolved inverse-frequency tables immediately follow
/// this POD header. Static algorithms have one table, LongRoPE has short then
/// long tables, and Dynamic NTK derives frequencies at invocation time.
struct RoPEF32KernelMetadata {
    int64_t head_dim{};
    int64_t rotary_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};
    int64_t original_context_length{};
    double theta{};
    double factor{};
    double beta_fast{};
    double beta_slow{};
    // Frozen resolver coefficient shared by every pair and invocation.
    double position_divisor{1.0};
    double rotary_output_scale{1.0};
    double low_freq_factor{};
    double high_freq_factor{};
    uint32_t freq_count{};
    bool truncate_correction_range{};
    uint8_t freq_table_count{};
    RoPEPairing pairing{RoPEPairing::kSplitHalf};
    RoPEAlgorithm algorithm{RoPEAlgorithm::kStandard};
};

/// Reads one frozen inverse frequency from the attrs byte blob.
///
/// Preconditions, established by the binding-time layout validation: attrs
/// holds the metadata header followed by freq_count * freq_table_count
/// doubles, with `table` and `pair` within range. Out-of-range reads are
/// detected only in debug builds (AM_DCHECK).
inline double ReadStaticInvFreqUnchecked(uint32_t freq_count,
                                         std::span<const std::byte> attrs,
                                         uint8_t table,
                                         int64_t pair) noexcept {
    double frequency = 0.0;
    const size_t index = static_cast<size_t>(table) * freq_count + static_cast<size_t>(pair);
    const size_t offset = sizeof(RoPEF32KernelMetadata) + index * sizeof(frequency);
    AM_DCHECK(offset <= attrs.size());
    AM_DCHECK(attrs.size() - offset >= sizeof(frequency));
    std::memcpy(&frequency, attrs.data() + offset, sizeof(frequency));
    return frequency;
}

/// @brief Pre-validated FP32 arguments for all supported RoPE algorithms.
///
/// All pointers and geometry are prepared once for a binding. `position_ids`
/// is read on each execution; its values are validated before the kernel
/// writes either output. The caller must not modify that buffer concurrently
/// with an invocation. Algorithm parameters and precomputed static frequency
/// tables are immutable attrs owned by the resolved kernel; the args retain no
/// borrowed vector pointers into those attrs.
struct RoPEF32KernelArgs {
    const float* q{};
    const float* k{};
    const int64_t* pos_ids{};
    float* q_output{};
    float* k_output{};

    int64_t seq_len{};
    int64_t head_dim{};
    int64_t rotary_dim{};
    int64_t num_q_heads{};
    int64_t num_kv_heads{};

    int64_t q_row_stride{};
    int64_t q_col_stride{1};
    int64_t k_row_stride{};
    int64_t k_col_stride{1};
    int64_t pos_stride{1};
    int64_t q_output_row_stride{};
    int64_t q_output_col_stride{1};
    int64_t k_output_row_stride{};
    int64_t k_output_col_stride{1};

    int64_t original_context_length{};
    double theta{};
    double factor{};
    // Copied from immutable attrs; position data remains invocation-local.
    double position_divisor{1.0};
    double rotary_output_scale{1.0};
    // Built by the cold-path params builder after it validates every static
    // frequency entry. LongRoPE uses the first value for its short table and
    // the second for its long table; other static algorithms use only short.
    double short_max_inv_freq{};
    double long_max_inv_freq{};
    uint32_t freq_count{};

    RoPEPairing pairing{RoPEPairing::kSplitHalf};
    RoPEAlgorithm algorithm{RoPEAlgorithm::kStandard};
};

/// @brief Runs the scalar FP32 RoPE reference kernel.
///
/// @param args Pre-validated tensor layout and frozen RoPE parameters.
/// @param attrs Kernel attrs.
/// @return InvalidArgument when a runtime position id is negative, Overflow
///         when its derived angle is not finite, or Ok after rotating both
///         outputs. A position failure occurs before any output write.
Status RunRoPEF32Reference(const RoPEF32KernelArgs& args,
                           std::span<const std::byte> attrs) noexcept;


} // namespace aethermind::cpu::detail

#endif // AETHERMIND_BACKEND_CPU_KERNELS_ROPE_ROPE_INTERNAL_H
