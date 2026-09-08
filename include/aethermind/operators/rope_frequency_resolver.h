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

struct ResolvedRoPEFrequencies {
    std::vector<double> inverse_frequencies;
    double attention_scale = 1.0;
};

/// Validates the algorithm-specific payload in addition to the scalar RoPE
/// geometry validated by InferRoPE.
Status ValidateRoPEFrequencyParameters(const RoPEParams& params);

/// Dynamic NTK and LongRoPE require an execution-time effective sequence
/// length. The remaining algorithms have a fixed frequency table.
bool IsDynamicRoPE(const RoPEAlgorithmParams& params) noexcept;

/// Computes the Dynamic NTK base without allocating a frequency table. This
/// supports backend reference kernels that derive one pair at a time.
StatusOr<double> ComputeDynamicNtkBase(double theta,
                                       int64_t rotary_dim,
                                       double factor,
                                       int64_t original_context_length,
                                       int64_t effective_sequence_length);

/// Resolves algorithms whose result is independent of position_ids.
/// Returns InvalidArgument for Dynamic NTK and LongRoPE.
StatusOr<ResolvedRoPEFrequencies> ResolveStaticRoPEFrequencies(const RoPEParams& params);

/// Resolves the frequency table for one execution. `effective_sequence_length`
/// is normally max(position_ids) + 1 and must be positive.
StatusOr<ResolvedRoPEFrequencies> ResolveDynamicRoPEFrequencies(
        const RoPEParams& params,
        int64_t effective_sequence_length);

} // namespace aethermind

#endif // AETHERMIND_OPERATORS_ROPE_FREQUENCY_RESOLVER_H
