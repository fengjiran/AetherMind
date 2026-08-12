#ifndef AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H
#define AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H

/// @file operator_inference.h
/// @brief Public dispatch and schema helpers for operator semantic inference.
///
/// `InferOperator()` is the public dispatch entry point. Operator-specific
/// functions in `aethermind::detail` are implementation details shared across
/// translation units.

#include "aethermind/operators/inference_result.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_schema.h"

namespace aethermind {

/// @brief Performs shape inference and constraint analysis for an operator.
///
/// @param op_type Operator type to infer.
/// @param params Typed parameter variant associated with `op_type`.
/// @param inputs Input tensor specifications in schema port order.
/// @return On success, an InferenceResult with inferred output specs and
///         deferred runtime checks. On failure, an error Status describing
///         the inference failure (e.g., kInvalidArgument for incompatible
///         input ranks, kUnimplemented for unsupported op types).
StatusOr<InferenceResult> InferOperator(OpType op_type,
                                        const OpParams& params,
                                        std::span<const TensorSpec> inputs);

/// @brief Validates that the input count matches the operator schema.
///
/// @param op_type Operator type whose schema defines the expected port count.
/// @param inputs Input tensor specifications to validate.
/// @return OkStatus on success; kInvalidArgument if the count mismatches.
Status ValidateInferenceInputCount(OpType op_type,
                                   std::span<const TensorSpec> inputs);

/// @brief Extracts the subset of input specs that contribute to tensor spec inference.
///
/// Filters `all_inputs` according to the `contributes_tensor_spec` flag on each
/// port in the operator schema. Ports with contributes_tensor_spec == false
/// (e.g., state inputs whose layout is determined by the operator itself) are
/// excluded from the result.
///
/// @param schema Operator schema defining port order and participation.
/// @param all_inputs Input tensor specifications indexed by schema port.
/// @return On success, a compacted vector containing only the input specs of
///         ports that contribute to tensor spec inference (may be empty).
///         On failure (input count mismatch or out-of-range port index),
///         an error Status.
/// @pre `all_inputs.size() == schema.input_ports.size()`.
StatusOr<std::vector<TensorSpec>> MakeCompactInputSpecs(
        const OperatorSchema& schema,
        std::span<const TensorSpec> all_inputs);

namespace detail {

StatusOr<InferenceResult> InferEmbedding(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferRmsNorm(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferLinear(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferQkvLinear(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferMatMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferRoPE(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferAttention(const OpParams& params, std::span<const TensorSpec> inputs);
/// @brief Infers KVCacheUpdate outputs (k_cache/v_cache forwarded verbatim).
///
/// Capacity contract, two layers:
///   * Graph level (necessary part): seq_len <= cache_len. Rejected statically
///     when both dims are static; deferred to the runtime when symbolic.
///   * Runtime level (sufficient part): current_pos + seq_len <= capacity,
///     enforced by KVCacheView::ValidateWrite. current_pos is session state
///     and never appears in the graph, so the graph cannot express the full
///     bound.
/// No deferred runtime checks are emitted: cache ports do not contribute to
/// tensor spec inference and capacity bounds are owned by the runtime layout.
StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSoftmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferArgmax(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferAdd(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSilu(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferSiluMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferElementwiseMul(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferReshape(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferPermute(const OpParams& params, std::span<const TensorSpec> inputs);
StatusOr<InferenceResult> InferReorder(const OpParams& params, std::span<const TensorSpec> inputs);

}// namespace detail

}// namespace aethermind

#endif// AETHERMIND_OPERATORS_OPERATOR_INFERENCE_H
