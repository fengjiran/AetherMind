#ifndef AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H
#define AETHERMIND_INFERENCE_MODEL_EXECUTION_POLICY_H

/// @file model_execution_policy.h
/// @brief Shared first-release FP32 admission and model-to-KV configuration.

#include "aethermind/base/status.h"
#include "aethermind/runtime/runtime_options.h"

#include <cstddef>

namespace aethermind {

class LoadedModel;
struct HfModelConfig;

/// @brief Applies the first-release scalar Float32 execution admission policy.
/// @param model Loaded model whose metadata and required resolved roles are read.
/// @return Ok, Unimplemented for unsupported precision, or an invalid metadata
///         error; diagnostic allocation failure returns empty ResourceExhausted.
/// @note Weight payloads are neither read nor converted and no borrow escapes.
///       Family and operator semantics remain the compiler's responsibility.
Status ValidateFp32ExecutionInputs(const LoadedModel& model);

/// @brief Validates actual required weights for explicit FP32 compatibility.
/// Accepts only scalar F32/F16/BF16 source weights and supported source hints.
/// Payload conversion remains the model normalizer's responsibility.
Status ValidateFp32CompatibleSourceInputs(const LoadedModel& model);

/// @brief Validates required execution weights as scalar FP32, ignoring hints.
/// Used after explicit normalization, where the original hint is provenance.
/// This does not validate tensor shapes, family semantics, or an existing Runtime.
Status ValidateFp32ExecutionWeights(const LoadedModel& model);

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
