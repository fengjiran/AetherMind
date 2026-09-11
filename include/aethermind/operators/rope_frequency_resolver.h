#ifndef AETHERMIND_OPERATORS_ROPE_FREQUENCY_RESOLVER_H
#define AETHERMIND_OPERATORS_ROPE_FREQUENCY_RESOLVER_H

/// @file rope_frequency_resolver.h
/// @brief Backend-independent RoPE frequency resolution.
///
/// The resolver owns algorithm-specific frequency math. Kernels consume its
/// inverse-frequency output and apply only pairing and tensor-layout logic.

#include "aethermind/base/status.h"
#include "aethermind/operators/op_params.h"

#include <cstdint>
#include <vector>

namespace aethermind {

/// @brief Resolved inverse frequencies for one RoPE execution.
///
/// Owns the per-pair frequency table consumed by kernels. The table length is
/// half the effective rotary dimension; kernels derive per-pair angles from it.
struct ResolvedRoPEFreqs {
    // One inverse frequency per rotary pair; size equals rotary_dim / 2.
    std::vector<double> inv_freqs;
    // Amplitude multiplier applied to the rotated output; 1.0 unless the
    // active YaRN or LongRoPE algorithm carries a front-end-normalized scale.
    double rotary_output_scale = 1.0;
};

/// @brief Validates the algorithm-specific payload of RoPE parameters.
///
/// @param params Semantic RoPE parameters, including the active frequency algorithm.
/// @return Ok when the algorithm payload satisfies its contract; InvalidArgument otherwise.
/// @note Scalar geometry is validated by InferRoPE; this function adds only the
///       per-algorithm checks (for example, finite positive factors and matching
///       LongRoPE factor-table lengths).
Status ValidateRoPEFreqParams(const RoPEParams& params);

/// @brief Reports whether an algorithm needs an execution-time sequence length.
///
/// @param params Active frequency algorithm variant.
/// @return True for Dynamic NTK and LongRoPE, false for the remaining algorithms.
bool IsDynamicRoPE(const RoPEAlgorithmParams& params) noexcept;

/// @brief Computes the Dynamic NTK base without allocating a frequency table.
///
/// @param theta Base frequency from the semantic RoPE parameters.
/// @param rotary_dim Effective rotary dimension; must exceed 2.
/// @param factor Dynamic NTK scaling factor; must be finite and positive.
/// @param original_context_len Training context length; must be positive.
/// @param effective_seq_len Execution-time length, normally max(position_ids) + 1.
/// @return Scaled theta base, or InvalidArgument for contract violations and Overflow
///         when the derived base is not finite.
/// @note Supports backend reference kernels that derive one pair at a time.
StatusOr<double> ComputeDynamicNtkBase(double theta,
                                       int64_t rotary_dim,
                                       double factor,
                                       int64_t original_context_len,
                                       int64_t effective_seq_len);

/// @brief Resolves frequencies for algorithms with a position-independent table.
///
/// @param params Semantic RoPE parameters carrying a static algorithm.
/// @return Owned frequency table, or InvalidArgument for Dynamic NTK and LongRoPE
///         which require an execution-time sequence length.
StatusOr<ResolvedRoPEFreqs> ResolveStaticRoPEFreqs(const RoPEParams& params);

/// @brief Resolves frequencies for algorithms that depend on the execution-time
///        sequence length.
///
/// @param params Semantic RoPE parameters carrying Dynamic NTK or LongRoPE.
/// @param effective_seq_len Execution-time length, normally max(position_ids) + 1;
///        must be positive.
/// @return Owned frequency table, or InvalidArgument for position-independent
///         algorithms (resolve those with ResolveStaticRoPEFreqs) and for a
///         non-positive effective_seq_len.
StatusOr<ResolvedRoPEFreqs> ResolveDynamicRoPEFreqs(
        const RoPEParams& params,
        int64_t effective_seq_len);

} // namespace aethermind

#endif // AETHERMIND_OPERATORS_ROPE_FREQUENCY_RESOLVER_H
