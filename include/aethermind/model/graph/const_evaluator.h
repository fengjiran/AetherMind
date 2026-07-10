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

/// Budget limits that control whether a constant-folding opportunity should
/// be materialized. Applied during Plan() — if an op exceeds the budget the
/// evaluator returns Unimplemented and the pass skips the node.
struct ConstEvalPolicy {
    /// Maximum total output bytes allowed for a single folded node.
    size_t max_output_bytes = size_t{64U} * 1024U;
    /// Maximum scalar element operations allowed for a single folded node.
    /// Complex evaluators may use this to estimate computation cost.
    size_t max_compute_elements = size_t{64U} * 1024U;
};

/// Description of one planned constant output produced by ConstEvaluator::Plan().
///
/// The evaluator decides how each output will be laid out, and the pass
/// allocates the contiguous buffer accordingly. `strides` and `nbytes` are
/// separate from `spec.shape` because the evaluator may request a non-contiguous
/// or broadcast-friendly layout. When `strides` is empty the pass falls back
/// to contiguous row-major strides derived from the static shape.
struct PlannedConstOutput {
    TensorSpec spec{};
    QuantizationSpec quantization{};
    /// Strides for the output buffer layout. Empty means contiguous row-major.
    std::vector<int64_t> strides{};
    /// Total byte size of the output buffer. Must match CountBytes(spec).
    size_t nbytes = 0;
    /// Debug name, convention: "folded_" + original output debug_name.
    /// Set by the evaluator during Plan(); used by the pass as the
    /// ConstantBinding name and AddConstant debug tag.
    std::string debug_name{};
};

/// Result of ConstEvaluator::Plan(). Contains one PlannedConstOutput per
/// graph output port.
struct ConstEvalPlan {
    std::vector<PlannedConstOutput> outputs{};
};

/// Abstract interface for compile-time constant evaluation of an operator.
///
/// The contract is two-phase: Plan() validates feasibility and describes
/// the expected output layout; Evaluate() writes the actual bytes.
///
/// Plan() must check all preconditions (dtype, shape, params, budget) and
/// return Unimplemented when the op cannot be folded. It must NOT allocate
/// output memory — that is the pass's responsibility.
///
/// Evaluate() must be stateless: it reads input TensorViews, writes into
/// output MutableTensorViews, and returns. It must not hold references to
/// the views or their data after returning.
///
/// All methods are thread-safe in the sense that the evaluator carries no
/// mutable state — implementations should be stateless singletons.
class ConstEvaluator {
public:
    virtual ~ConstEvaluator() = default;

    /// Validates folding feasibility and describes the expected output layout.
    ///
    /// @param inputs  Descriptors of the node's input values.
    /// @param outputs Descriptors of the node's output values.
    /// @param params  Operator-specific parameters.
    /// @param policy  Budget constraints for the folding decision.
    /// @return ConstEvalPlan with outputs populated, or Unimplemented when
    ///         the op cannot be folded under the given constraints.
    /// @pre  All `inputs` must have ConstantValue payloads with inline_data.
    /// @post outputs.size() == outputs.size() from the caller.
    AM_NODISCARD virtual StatusOr<ConstEvalPlan> Plan(
            std::span<const NodeOutputDesc> inputs,
            std::span<const NodeOutputDesc> outputs,
            const OpParams& params,
            const ConstEvalPolicy& policy) const = 0;

    /// Writes the folded output bytes for a previously planned operation.
    ///
    /// @param inputs  Read-only views into the constant input buffers.
    /// @param outputs Writable views into pre-allocated output buffers.
    /// @param params  Same operator parameters passed to Plan().
    /// @pre  `inputs` and `outputs` sizes, dtypes, and layouts must match
    ///       the corresponding descriptors from Plan().
    /// @post Output buffers contain the folded result.
    /// @note Do not save pointers to the views or their data.
    AM_NODISCARD virtual Status Evaluate(std::span<const TensorView> inputs,
                                         std::span<MutableTensorView> outputs,
                                         const OpParams& params) const = 0;
};

/// Returns the ConstEvaluator registered for `op_type`, or nullptr if
/// no compile-time evaluator is available for this op.
/// The lookup table is a static readonly structure — no mutable global state.
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
/// Returns kOverflow when stride multiplication overflows int64_t.
AM_NODISCARD StatusOr<std::vector<int64_t>> MakeContiguousStrides(std::span<const int64_t> shape);
}// namespace aethermind

#endif
