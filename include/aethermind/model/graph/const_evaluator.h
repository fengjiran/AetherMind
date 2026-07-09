#ifndef AETHERMIND_MODEL_GRAPH_CONST_EVALUATOR_H
#define AETHERMIND_MODEL_GRAPH_CONST_EVALUATOR_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/model/graph/graph_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace aethermind {

struct ConstEvalPolicy {
    size_t max_output_bytes = size_t{64U} * 1024U;
    size_t max_compute_elements = size_t{64U} * 1024U;
};

struct PlannedConstOutput {
    TensorSpec spec{};
    QuantizationSpec quantization{};
    std::vector<int64_t> strides{};
    size_t nbytes = 0;
    std::string debug_name{};
};

struct ConstEvalPlan {
    std::vector<PlannedConstOutput> outputs{};
};

class ConstEvaluator {
public:
    virtual ~ConstEvaluator() = default;

    AM_NODISCARD virtual StatusOr<ConstEvalPlan> Plan(std::span<const NodeOutputDesc> inputs,
                                                      std::span<const NodeOutputDesc> outputs,
                                                      const OpParams& params,
                                                      const ConstEvalPolicy& policy) const = 0;

    AM_NODISCARD virtual Status Evaluate(std::span<const TensorView> inputs,
                                         std::span<MutableTensorView> outputs,
                                         const OpParams& params) const = 0;
};

AM_NODISCARD const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept;

// ── Shape/stride helpers shared by constant evaluator and folding pass ──

/// Extracts a concrete static shape from a TensorSpec.
/// Returns Unimplemented if the shape is dynamic or unranked.
AM_NODISCARD StatusOr<std::vector<int64_t>> ExtractStaticShape(const TensorSpec& spec);

/// Counts the total number of elements implied by a static shape.
AM_NODISCARD StatusOr<int64_t> CountElements(std::span<const int64_t> shape);

/// Computes the byte size of a tensor given its spec (requires static shape).
AM_NODISCARD StatusOr<size_t> CountBytes(const TensorSpec& spec);

/// Builds contiguous (row-major) strides from a static shape.
AM_NODISCARD std::vector<int64_t> MakeContiguousStrides(std::span<const int64_t> shape);

/// Returns true when both specs have the same static shape.
AM_NODISCARD bool SameStaticShape(const TensorSpec& lhs, const TensorSpec& rhs);

}// namespace aethermind

#endif
