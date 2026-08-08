#ifndef AETHERMIND_OPERATORS_INFERENCE_RESULT_H
#define AETHERMIND_OPERATORS_INFERENCE_RESULT_H

/// @file inference_result.h
/// @brief Output specifications and deferred checks produced by operator inference.

#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <vector>

namespace aethermind {

/// @brief Result of operator-level tensor-spec inference.
///
/// `outputs` carries the inferred output tensor specs. `runtime_checks` carries
/// shape constraints that cannot be fully proven until concrete runtime shapes
/// are known.
struct InferenceResult {
    std::vector<TensorSpec> outputs{};
    std::vector<ShapeConstraint> runtime_checks{};
};

}// namespace aethermind

#endif// AETHERMIND_OPERATORS_INFERENCE_RESULT_H
