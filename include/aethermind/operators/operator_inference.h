/// Operator semantic layer: validation, shape inference, and input filtering.
///
/// This header is the single public entry point for operator-level semantic
/// analysis used during graph construction and workspace planning. It exposes
/// three categories of API:
///
///   * ValidateOperatorParams    - parameter validation per OpType
///   * InferOperator             - shape inference dispatch (entry point)
///   * MakeCompactInputSpecs    - input filtering by schema port flags
///
/// The Infer* free functions in the aethermind::detail namespace are not part
/// of the public API; InferOperator dispatches to the Infer* function matching
/// the OpType.
#ifndef AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H
#define AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/operators/inference_result.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "macros.h"

#include <span>
#include <string_view>
#include <vector>

namespace aethermind {

/// Validates that the operator parameters are consistent with the operator type.
///
/// Checks that the OpParams variant holds the expected alternative for
/// op_type and that all parameter values are within valid ranges (e.g.,
/// epsilon > 0 for RmsNorm, non-negative dimensions for Attention).
///
/// \param op_type  The operator type to validate against.
/// \param params   The operator parameters to validate.
/// \return Ok if the parameters are valid; otherwise an error Status
///         (typically kInvalidArgument) describing the inconsistency.
AM_NODISCARD Status ValidateOperatorParams(OpType op_type, const OpParams& params);

/// Performs shape inference and constraint analysis for an operator.
///
/// Given the operator type, its parameters, and the input tensor specs,
/// computes the inferred output tensor specs and any deferred runtime shape
/// constraints that cannot be fully proven without concrete runtime shapes.
///
/// This is the primary entry point for operator-level semantic analysis used
/// during graph construction and workspace planning.
///
/// \param op_type  The operator type.
/// \param params   The operator parameters (must pass ValidateOperatorParams first).
/// \param inputs   The input tensor specs for the operator.
/// \return On success, an InferenceResult with inferred output specs and
///         deferred runtime checks. On failure, an error Status describing
///         the inference failure (e.g., kInvalidArgument for incompatible
///         input ranks, kUnimplemented for unsupported op types).
AM_NODISCARD StatusOr<InferenceResult> InferOperator(OpType op_type,
                                                     const OpParams& params,
                                                     std::span<const TensorSpec> inputs);

/// Extracts the subset of input specs that contribute to tensor spec inference.
///
/// Filters all_inputs according to the contributes_tensor_spec flag on each
/// port in the operator schema. Ports with contributes_tensor_spec == false
/// (e.g., state inputs whose layout is determined by the operator itself) are
/// excluded from the result.
///
/// Validates that all_inputs.size() matches schema.input_ports.size() and
/// that every port index is within bounds before accessing the span.
///
/// \param schema      The operator schema defining the port layout.
/// \param all_inputs  All input tensor specs for the operator, indexed by port
///                    position.
/// \return On success, a compacted vector containing only the input specs of
///         ports that contribute to tensor spec inference. The returned vector
///         may be empty if no ports contribute. On failure (input count
///         mismatch or out-of-range port index), an error Status.
AM_NODISCARD StatusOr<std::vector<TensorSpec>> MakeCompactInputSpecs(
        const OperatorSchema& schema,
        std::span<const TensorSpec> all_inputs);

namespace detail {

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

// Shared broadcast-binary inference helper used by SiluMul and ElementwiseMul
// (ops whose semantics are elementwise binary with broadcast). Defined in
// operator_inference.cpp alongside the dispatch hub.
StatusOr<InferenceResult> InferBroadcastBinary(const OpParams& params,
                                               std::span<const TensorSpec> inputs,
                                               std::string_view op_name);

}// namespace detail

}// namespace aethermind

#endif// AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H
