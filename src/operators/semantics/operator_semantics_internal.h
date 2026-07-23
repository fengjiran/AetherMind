/// Internal analysis function declarations for operator semantics.
///
/// Each OpType has a corresponding Infer* function that performs
/// operator-specific shape inference. These are declared in the
/// aethermind::detail namespace and are not part of the public API.

#ifndef AETHERMIND_OPERATORS_SEMANTICS_OPERATOR_SEMANTICS_INTERNAL_H
#define AETHERMIND_OPERATORS_SEMANTICS_OPERATOR_SEMANTICS_INTERNAL_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_semantics.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <span>

namespace aethermind::detail {

StatusOr<InferenceResult> InferEmbedding(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferRmsNorm(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferLinear(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferMatMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferRoPE(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferAttention(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSoftmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferArgmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferAdd(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSilu(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSiluMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferElementwiseMul(const OpParams& params, std::span<const TensorSpec> inputs);

}// namespace aethermind::detail

#endif// AETHERMIND_OPERATORS_SEMANTICS_OPERATOR_SEMANTICS_INTERNAL_H
