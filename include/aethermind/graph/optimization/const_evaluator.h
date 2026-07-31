#ifndef AETHERMIND_GRAPH_OPTIMIZATION_CONST_EVALUATOR_H
#define AETHERMIND_GRAPH_OPTIMIZATION_CONST_EVALUATOR_H

/// @file const_evaluator.h
/// @brief Compile-time constant evaluation interface for constant folding.
///
/// Defines the ConstEvaluator two-phase Plan/Evaluate contract, the
/// ConstEvalPolicy budget, and shared shape/stride helpers used by the
/// constant folding pass.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/graph/graph_types.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace aethermind {

/// @brief Budget limits that control whether a constant-folding opportunity should
/// be materialized. Applied during Plan() — if an op exceeds the budget the
/// evaluator returns Unimplemented and the pass skips the node.
struct ConstEvalPolicy {
    /// @brief Maximum total output bytes allowed for a single folded node.
    size_t max_output_bytes = size_t{64U} * 1024U;
    /// @brief Maximum scalar element operations allowed for a single folded node.
    ///        Complex evaluators may use this to estimate computation cost.
    size_t max_compute_elements = size_t{64U} * 1024U;
};

/// @brief Description of one planned constant output produced by ConstEvaluator::Plan().
///
/// The evaluator decides how each output will be laid out, and the pass
/// allocates the contiguous buffer accordingly. `strides` and `nbytes` are
/// separate from `spec.shape` because the evaluator may request a non-contiguous
/// or broadcast-friendly layout. When `strides` is empty the pass falls back
/// to contiguous row-major strides derived from the static shape.
struct PlannedConstOutput {
    TensorSpec spec{};
    QuantizationSpec quantization{};
    /// @brief Strides for the output buffer layout. Empty means contiguous row-major.
    std::vector<int64_t> strides{};
    /// @brief Total byte size of the output buffer. Must match CountBytes(spec).
    size_t nbytes = 0;
    /// @brief Debug name, convention: "folded_" + original output debug_name.
    ///        Set by the evaluator during Plan(); used by the pass as the
    ///        ConstantBinding name and AddConstant debug tag.
    std::string debug_name{};
};

/// @brief Result of ConstEvaluator::Plan(). Contains one PlannedConstOutput per
/// graph output port.
struct ConstEvalPlan {
    std::vector<PlannedConstOutput> outputs{};
};

/// @brief Abstract interface for compile-time constant evaluation of an operator.
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

    /// @brief Validates folding feasibility and describes the expected output layout.
    ///
    /// @param inputs  Descriptors of the node's input values (spec-bearing).
    /// @param outputs Descriptors of the node's output values (spec-bearing).
    /// @param params  Operator-specific parameters.
    /// @param policy  Budget constraints for the folding decision.
    /// @return ConstEvalPlan with outputs populated, or Unimplemented when
    ///         the op cannot be folded under the given constraints.
    /// @pre  All `inputs` must have ConstantValue payloads with inline_data.
    /// @post outputs.size() == outputs.size() from the caller.
    AM_NODISCARD virtual StatusOr<ConstEvalPlan> Plan(
            std::span<const GraphValueDesc> inputs,
            std::span<const GraphValueDesc> outputs,
            const OpParams& params,
            const ConstEvalPolicy& policy) const = 0;

    /// @brief Writes the folded output bytes for a previously planned operation.
    ///
    /// @param inputs  Read-only views into the constant input buffers.
    /// @param outputs Writable views into pre-allocated output buffers.
    /// @param params  Same operator parameters passed to Plan().
    /// @return Status::Ok() on success, or the first error encountered.
    /// @pre  `inputs` and `outputs` sizes, dtypes, and layouts must match
    ///       the corresponding descriptors from Plan().
    /// @post Output buffers contain the folded result.
    /// @note Do not save pointers to the views or their data.
    AM_NODISCARD virtual Status Evaluate(std::span<const TensorView> inputs,
                                         std::span<MutableTensorView> outputs,
                                         const OpParams& params) const = 0;
};

/// @brief Returns the ConstEvaluator registered for `op_type`, or nullptr if
/// no compile-time evaluator is available for this op.
/// The lookup table is a static readonly structure — no mutable global state.
/// @param op_type Operator type to look up.
/// @return Registered evaluator pointer, or nullptr when none is registered.
AM_NODISCARD const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept;

// ── Shape/stride helpers shared by constant evaluator and folding pass ──

/// @brief Extracts a concrete static shape from a TensorSpec.
/// @param spec Tensor spec to inspect.
/// @return Static shape as a vector of dimensions, or Unimplemented if the
///         shape is dynamic or unranked.
AM_NODISCARD StatusOr<std::vector<int64_t>> ExtractStaticShape(const TensorSpec& spec);

/// @brief Counts the total number of elements implied by a static shape.
/// @param shape Static shape to size.
/// @return Total element count, or kOverflow on int64_t overflow.
AM_NODISCARD StatusOr<int64_t> CountElements(std::span<const int64_t> shape);

/// @brief Computes the byte size of a tensor given its spec (requires static shape).
/// @param spec Tensor spec describing dtype and shape.
/// @return Total byte size, or Unimplemented if the shape is dynamic/unranked,
///         or kOverflow on size multiplication overflow.
AM_NODISCARD StatusOr<size_t> CountBytes(const TensorSpec& spec);

/// @brief Builds contiguous (row-major) strides from a static shape.
/// @param shape Static shape to derive strides for.
/// @return Row-major strides, or kOverflow when stride multiplication
///         overflows int64_t.
AM_NODISCARD StatusOr<std::vector<int64_t>> MakeContiguousStrides(std::span<const int64_t> shape);
}// namespace aethermind

#endif
