#ifndef AETHERMIND_COMPILER_LOWERED_GRAPH_H
#define AETHERMIND_COMPILER_LOWERED_GRAPH_H

/// @file lowered_graph.h
/// @brief Immutable compiler artifact between semantic graph and execution planning.

#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/workspace_types.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/operators/op_params.h"

#include <span>
#include <string>
#include <vector>

namespace aethermind {

struct GraphLoweringConfig;
class ModelGraph;

/// @brief Graph values bound to a lowered step in semantic schema-port order.
///
/// State ports remain in these vectors even when they are omitted from the
/// compact tensor-spec view consumed by runtime kernels.
struct LoweredStepBinding {
    /// Originating semantic node; each node is lowered into at most one step.
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
};

/// @brief Unresolved must-alias relation in semantic schema-port coordinates.
///
/// Execution converts verified records into StateAliasPlan; this compiler
/// artifact never includes execution runtime types.
struct LoweredStateAlias {
    /// Index into LoweredGraph::steps().
    size_t step_index = 0;
    uint32_t input_port = 0;
    uint32_t output_port = 0;
    GraphValueId input{};
    GraphValueId output{};
};

/// @brief Semantic metadata emitted by graph lowering after ModelGraph
/// validation.
///
/// This is deliberately distinct from execution::ExecutionPlanNodeSpec: the
/// latter is an untrusted raw execution request and is revalidated with
/// InferOperator. LoweredNodeSpec preserves the already-validated semantic
/// result and is only reachable through a finalized LoweredGraph.
struct LoweredNodeSpec {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
    WorkspaceRequirement workspace_requirement{};
    /// Complete schema-port-ordered specs. State ports remain present even
    /// when they do not contribute to compact runtime tensor bindings.
    std::vector<TensorSpec> input_specs{};
    std::vector<TensorSpec> output_specs{};
    /// Deferred shape constraints copied from semantic inference; they cannot
    /// be proven until concrete runtime shapes are known, so execution
    /// re-validates them before running the kernel.
    std::vector<ShapeConstraint> runtime_checks{};
    OpParams op_params{};
};

/// @brief A lowered node spec paired with the value binding of the same node.
struct LoweredStep {
    LoweredNodeSpec spec{};
    LoweredStepBinding binding{};
};

/// @brief Value metadata owned by the compiler artifact after ModelGraph
/// lifetime ends.
///
/// The entry at index id.index describes GraphValueId id.
struct LoweredValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{};
    QuantizationSpec quantization{};
    std::string name{};
};

/// @brief A finalized, structurally verified compiler artifact.
///
/// Its storage is private and exposed only through const views. The sole
/// construction path is LowerModelGraph, which validates both semantic graph
/// input and the finalized artifact structure. Execution may trust its
/// inference metadata without re-running InferOperator, while raw execution
/// node requests continue to use ExecutionPlanNodeSpec and validation.
///
/// @note Immutable after construction, so concurrent reads are safe. All
/// accessors return spans that borrow from this object and must not outlive
/// it.
class LoweredGraph {
public:
    LoweredGraph() = default;

    AM_NODISCARD std::span<const LoweredStep> steps() const noexcept {
        return steps_;
    }

    AM_NODISCARD std::span<const LoweredValueDesc> values() const noexcept {
        return values_;
    }

    AM_NODISCARD std::span<const GraphValueId> model_inputs() const noexcept {
        return model_inputs_;
    }

    AM_NODISCARD std::span<const GraphValueId> model_outputs() const noexcept {
        return model_outputs_;
    }

    AM_NODISCARD std::span<const LoweredStateAlias> state_aliases() const noexcept {
        return state_aliases_;
    }

    AM_NODISCARD size_t size() const noexcept {
        return steps_.size();
    }

    /// @brief Mutable construction state for a LoweredGraph.
    ///
    /// The sole construction path is `LowerModelGraph`, which accumulates
    /// steps/values/IO/aliases here and freezes them via Build(). It is also
    /// the focused test seam for malformed artifact input. Defined below so
    /// it can reference the complete LoweredGraph.
    class Builder;

private:
    std::vector<LoweredStep> steps_{};
    std::vector<LoweredValueDesc> values_{};
    std::vector<GraphValueId> model_inputs_{};
    std::vector<GraphValueId> model_outputs_{};
    std::vector<LoweredStateAlias> state_aliases_{};
};

/// @brief Mutable construction state for a LoweredGraph (see
/// LoweredGraph::Builder).
///
/// Nested in LoweredGraph so construction has direct access to the private
/// storage without a friend declaration. Build() validates the complete
/// structure before freezing it into the immutable artifact.
class LoweredGraph::Builder {
public:
    std::vector<LoweredStep> steps{};
    std::vector<LoweredValueDesc> values{};
    std::vector<GraphValueId> model_inputs{};
    std::vector<GraphValueId> model_outputs{};
    std::vector<LoweredStateAlias> state_aliases{};

    /// Validates the accumulated structure without consuming the builder.
    Status Validate() const;

    /// Validates and moves the accumulated state into a frozen LoweredGraph,
    /// consuming the builder.
    StatusOr<LoweredGraph> Build() &&;
};

/// @brief Validates structural and provenance invariants of a finalized
/// compiler artifact without re-running semantic inference.
///
/// This intentionally checks IDs, schema arity, bindings, copied specs,
/// selector dtypes, model I/O, and declared state aliases rather than
/// creating a second semantic authority.
///
/// @param lowered Finalized graph to validate.
/// @return Ok on success; otherwise Internal with a message describing the
///         first violated invariant.
Status ValidateLoweredGraph(const LoweredGraph& lowered);

}// namespace aethermind

#endif
