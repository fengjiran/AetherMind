#ifndef AETHERMIND_MODEL_WEIGHT_RAW_WEIGHT_FP32_CONVERSION_H
#define AETHERMIND_MODEL_WEIGHT_RAW_WEIGHT_FP32_CONVERSION_H

/// @file raw_weight_fp32_conversion.h
/// @brief Explicit owned FP32 logical-weight conversion before compilation.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/bf16_conversion.h"
#include "aethermind/dtypes/fp16_conversion.h"
#include "aethermind/model/weight/raw_weight_storage_usage.h"

#include <cstddef>
#include <memory>

namespace aethermind {

class Allocator;
class LoadedModel;

/// @brief Final retained owned-weight payload cap; zero means unlimited.
/// Mapped/external backing, temporary source backing, and metadata are excluded.
/// Both callbacks must be non-null, even if their source dtype is absent.
struct RawWeightFp32ConversionOptions {
    size_t max_owned_weight_bytes = 0;
    /// Non-null host widening callback; must obey Fp16ConversionFunc's contract.
    /// Borrowed for this call only, invoked once per unique F16 source tensor.
    Fp16ConversionFunc fp16_to_fp32 = ConvertFp16ToFp32Scalar;
    /// Non-null host widening callback; must obey Bf16ConversionFunc's contract.
    /// Borrowed for this call only, invoked once per unique BF16 source tensor.
    Bf16ConversionFunc bf16_to_fp32 = ConvertBf16ToFp32Scalar;
};

/// @brief Visible weight storage observed during one conversion operation.
///
/// This is metadata only: it does not retain checkpoint owners or measure RSS.
/// Conversion totals count identical source views once per ownership identity.
/// The peak covers source owned backing plus converted payload while both live;
/// it excludes subsequent compilation/packing and allocator overhead.
struct RawWeightFp32ConversionUsage {
    RawWeightStorageUsage source_weights{};
    RawWeightStorageUsage converted_weights{};
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
StatusOr<std::unique_ptr<LoadedModel>> ConvertRawModelWeightsToFp32(
        const LoadedModel& model,
        Allocator& allocator,
        const RawWeightFp32ConversionOptions& options = {});

} // namespace aethermind

#endif
