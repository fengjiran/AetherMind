#ifndef AETHERMIND_MODEL_WEIGHT_MODEL_WEIGHT_NORMALIZATION_H
#define AETHERMIND_MODEL_WEIGHT_MODEL_WEIGHT_NORMALIZATION_H

/// @file model_weight_normalization.h
/// @brief Explicit owned FP32 logical-weight normalization before compilation.

#include "aethermind/base/status.h"
#include "aethermind/model/weight/model_weight_storage_summary.h"

#include <cstddef>
#include <memory>

namespace aethermind {

class Allocator;
class LoadedModel;

/// @brief Final retained owned-weight payload cap; zero means unlimited.
/// Mapped/external backing, temporary source backing, and metadata are excluded.
struct ModelWeightNormalizationOptions {
    size_t max_owned_weight_bytes = 0;
};

/// @brief Visible weight storage observed during one normalization operation.
///
/// This is metadata only: it does not retain checkpoint owners or measure RSS.
/// Conversion totals count identical source views once per ownership identity.
/// The peak covers source owned backing plus converted payload while both live;
/// it excludes subsequent compilation/packing and allocator overhead.
struct ModelWeightNormalizationUsage {
    ModelWeightStorageSummary source_weights{};
    ModelWeightStorageSummary normalized_weights{};
    size_t converted_source_bytes = 0;
    size_t converted_weight_bytes = 0;
    size_t peak_owned_weight_bytes = 0;
    bool peak_owned_weight_bytes_complete = true;
};

/// @brief Produces a new model with actual F16/BF16 weights widened to FP32.
///
/// All populated required roles must be valid contiguous scalar F32/F16/BF16
/// views. F32 views share their original owners; identical reduced-precision
/// views share one converted owner. Configuration, including source dtype hints,
/// remains unchanged. Missing optional lm_head remains missing.
///
/// Conversion preserves finite values exactly, including subnormals and signed
/// zero. NaNs retain sign and payload with the FP32 quiet bit set. Source bytes
/// may be unaligned and are never modified. Destination payloads use the supplied
/// CPU allocator and own their storage independently of its lifetime.
///
/// All source metadata validation, size accounting, and cap checks precede payload
/// allocation. Failure leaves the input model unchanged and releases candidates.
/// Unknown retained F32 backing prevents a finite owned-payload cap; unknown
/// source backing otherwise remains explicit in usage completeness flags.
/// @return New owning model, or a validation/overflow/allocation error.
AM_NODISCARD StatusOr<std::unique_ptr<LoadedModel>> NormalizeLoadedModelWeightsToFp32(
        const LoadedModel& model,
        Allocator& allocator,
        const ModelWeightNormalizationOptions& options = {});

} // namespace aethermind

#endif
