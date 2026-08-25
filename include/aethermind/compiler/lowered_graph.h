#ifndef AETHERMIND_COMPILER_LOWERED_GRAPH_H
#define AETHERMIND_COMPILER_LOWERED_GRAPH_H

/// @file lowered_graph.h
/// @brief Immutable compiler artifact between semantic graph and execution planning.
///
/// Type layout by responsibility domain:
///   - step domain:  LoweredStepSpec / LoweredStepBinding / LoweredStep
///   - graph domain: LoweredStateAlias / LoweredValueDesc
///   - container:    LoweredGraph (+ nested Builder)

#include "aethermind/base/kernel_selector.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/operators/op_params.h"

#include <span>
#include <string>
#include <vector>

namespace aethermind {

struct GraphLoweringConfig;
class ModelGraph;

// ── Step execution spec ───────────────────────────────────────────

/// @brief Semantic metadata emitted by graph lowering after ModelGraph
/// validation.
///
/// This is deliberately distinct from execution::ExecutionPlanNodeSpec: the
/// latter is an untrusted raw execution request and is revalidated with
/// InferOperator. LoweredStepSpec preserves the already-validated semantic
/// result and is only reachable through a finalized LoweredGraph.
///
/// This is the sole consumption surface of ExecutionPlanBuilder (the trusted
/// path projects it out of LoweredStep; the untrusted path uses the
/// isomorphic ExecutionPlanNodeSpec), so it stays free of graph-coordinate
/// types.
struct LoweredStepSpec {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
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

// ── Step graph binding ────────────────────────────────────────────

/// @brief Graph values bound to a lowered step in semantic schema-port order.
///
/// State ports remain in these vectors even when they are omitted from the
/// compact tensor-spec view consumed by runtime kernels. Consumers: the P2
/// runtime tensor/state binding wires the graph-value identity kept here;
/// ValidateLoweredGraph checks these ids against the value table.
struct LoweredStepBinding {
    /// Originating semantic node; each node is lowered into at most one step.
    GraphNodeId node{};
    std::vector<GraphValueId> input_values{};
    std::vector<GraphValueId> output_values{};
};

// ── Step pair ─────────────────────────────────────────────────────

/// @brief A lowered step spec paired with the value binding of the same node.
///
/// The 1:1 pairing is a type-level invariant: spec and binding are moved and
/// validated together and can never drift apart.
struct LoweredStep {
    LoweredStepSpec spec{};
    LoweredStepBinding binding{};
};

// ── State aliases (parallel to steps) ─────────────────────────────

/// @brief Unresolved must-alias relation in semantic schema-port coordinates.
///
/// Execution converts verified records into StateAliasPlan; this compiler
/// artifact never includes execution runtime types. Records are declared by
/// OperatorSchema::state_alias_ports, recorded by LowerModelGraph with
/// known step/port coordinates, and validated by ValidateLoweredGraph.
struct LoweredStateAlias {
    /// Index into LoweredGraph::steps().
    size_t step_index = 0;
    uint32_t input_port = 0;
    uint32_t output_port = 0;
    GraphValueId input{};
    GraphValueId output{};
};

// ── Value metadata table (parallel to steps) ──────────────────────

/// @brief Value metadata owned by the compiler artifact after ModelGraph
/// lifetime ends.
///
/// The entry at index id.index describes GraphValueId id. Consumers: the P2
/// binding wiring resolves payloads (weight/state/constant) by value id;
/// debugging and dumps read spec/name here.
struct LoweredValueDesc {
    TensorSpec spec{};
    GraphValuePayload payload{};
    QuantizationSpec quantization{};
    std::string name{};
};

// ── Immutable container ───────────────────────────────────────────

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

    /// @brief Instance identity of this compiler artifact.
    ///
    /// Assigned once at Build(); used to tie packed-weight artifacts and their
    /// PackedWeightStore to the exact lowered graph they were produced for.
    AM_NODISCARD uint64_t artifact_id() const noexcept {
        return artifact_id_;
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
    uint64_t artifact_id_ = 0;
};

/// @brief Mutable construction state for a LoweredGraph (see
/// LoweredGraph::Builder).
///
/// Nested in LoweredGraph so construction has direct access to the private
/// storage without a friend declaration. Build() validates the complete
/// structure before freezing it into the immutable artifact. Its fields
/// mirror LoweredGraph's private storage one-to-one: Build() && moves them
/// in, so the two field lists stay identical by construction.
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
