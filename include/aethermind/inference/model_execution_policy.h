#ifndef AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H
#define AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H

/// @file model_execution_policy.h
/// @brief Shared first-release FP32 admission and model-to-KV configuration.

#include "aethermind/base/status.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/runtime/runtime_options.h"

#include <cstddef>

namespace aethermind {

class LoadedModel;
struct HfModelConfig;

/// @brief Checks source dtype hints and actual required weights for a policy.
/// @return Ok, Unimplemented for unsupported source precision, or an invalid
///         metadata/policy error; diagnostic allocation failure returns empty
///         ResourceExhausted. Payloads are neither read nor converted.
/// @note Model family, shapes and operator semantics remain compiler concerns.
Status ValidateSourceWeights(const LoadedModel& model, WeightPolicy policy);

/// @brief Maps validated model geometry and explicit capacity to FP32 KV options.
/// @param config Configuration already validated by ModelLoader.
/// @param context_tokens Positive KV capacity no larger than the model context.
/// @return Owning KV options with logical head dimension, scalar Float32 and
///         64-byte alignment, or a conversion/geometry/capacity error; diagnostic
///         allocation failure returns empty ResourceExhausted.
/// @note Reads metadata only. It does not allocate storage, build a Runtime,
///       inspect backends, or validate an existing Runtime/model pair. Only an
///       absent head_dim (zero) falls back to exact hidden_size/head-count division.
StatusOr<KVCacheRuntimeOptions> MakeFp32KVCacheOptions(
        const HfModelConfig& config, size_t context_tokens);

} // namespace aethermind

#endif
