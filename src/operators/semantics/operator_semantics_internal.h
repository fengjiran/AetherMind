/// Internal analysis function declarations for operator semantics.
///
/// Each OpType has a corresponding Analyze* function that performs
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

StatusOr<InferenceResult> AnalyzeEmbedding(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeRmsNorm(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeLinear(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeMatMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeRoPE(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeAttention(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeKVCacheUpdate(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeSoftmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeArgmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeAdd(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeSilu(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeSiluMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> AnalyzeElementwiseMul(const OpParams& params, std::span<const TensorSpec> inputs);

}// namespace aethermind::detail

#endif// AETHERMIND_OPERATORS_SEMANTICS_OPERATOR_SEMANTICS_INTERNAL_H
