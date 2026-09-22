#ifndef AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H
#define AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H

/// @file executable_model.h
/// @brief Production-ready model prepared from a compiler artifact.

#include "aethermind/base/kernel_attrs.h"
#include "aethermind/base/status.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/inference/weight_binding_storage.h"
#include "aethermind/model/packed_weight_store.h"

#include <cstdint>

namespace aethermind {

class Runtime;

/// @brief Immutable, execution-ready model owned between compilation and sessions.
///
/// Owns everything a session needs to specialize and run the model: the compiler
/// artifact and through it the raw weight backing, the packed-weight artifacts,
/// the metadata that the immutable binding table borrows, that table, and the
/// execution plan. Sessions never read the compiler artifact, resolve weights by
/// role, or interpret packing decisions — preparation has already done it.
///
/// Lifetime, outermost first:
/// - the Runtime passed to PrepareExecutableModel must outlive this object,
///   because the plan's resolved kernels borrow its backends;
/// - this object must outlive every PreparedExecutionBindings built from its plan
///   and bindings, because prepared bindings borrow the weight data pointers
///   recorded here;
/// - member declaration order is the teardown contract: destruction runs in
///   reverse, releasing the plan and the binding table before the metadata and
///   the artifact backing they borrow.
///
/// Move-only. Not thread-safe: preparation is a cold-path operation and callers
/// must serialize concurrent access.
class ExecutableModel {
public:
    ExecutableModel(ExecutableModel&&) noexcept = default;
    ExecutableModel& operator=(ExecutableModel&&) noexcept = default;
    ExecutableModel(const ExecutableModel&) = delete;
    ExecutableModel& operator=(const ExecutableModel&) = delete;
    ~ExecutableModel() = default;

    /// @brief Returns the plan to run for `phase`.
    ///
    /// Prefill and decode currently share one immutable plan; the query still
    /// checks the phase the artifact was compiled for, so a phase-specific
    /// artifact cannot be silently reused for the other phase.
    ///
    /// @param phase Phase the caller intends to run.
    /// @return Borrowed plan owned by this model, or an error on phase mismatch.
    AM_NODISCARD StatusOr<const ExecutionPlan*> plan(ExecPhase phase) const noexcept;

    /// @brief Returns the immutable weight and constant bindings for `phase`.
    ///
    /// Model inputs are deliberately absent: a session appends its own token and
    /// position bindings before calling PrepareExecutionBindings.
    ///
    /// @param phase Phase the caller intends to run.
    /// @return Borrowed bindings owned by this model, or an error on phase
    ///         mismatch.
    AM_NODISCARD StatusOr<const ExternalTensorBindings*> immutable_weight_bindings(
            ExecPhase phase) const noexcept;

    /// @brief Returns the identity of the compiled artifact this model was built
    ///        from, matching the packed-weight store's source id.
    AM_NODISCARD uint64_t artifact_id() const noexcept;

    /// @brief Returns the single phase this artifact was compiled for.
    AM_NODISCARD ExecPhase phase() const noexcept;

private:
    friend StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                            LoweredModelArtifact artifact);

    ExecutableModel(LoweredModelArtifact artifact,
                    PackedWeightStore packed_weights,
                    WeightBindingStorage binding_storage,
                    ExternalTensorBindings bindings,
                    ExecutionPlan plan,
                    ExecPhase phase) noexcept;

    /// @brief Validates a phase query against the compiled artifact phase.
    AM_NODISCARD Status CheckPhase(ExecPhase phase) const noexcept;

    LoweredModelArtifact artifact_{};
    PackedWeightStore packed_weights_{};
    WeightBindingStorage binding_storage_{};
    ExternalTensorBindings bindings_{};
    ExecutionPlan plan_{};
    ExecPhase phase_ = ExecPhase::kBoth;
};

/// @brief Prepares a compiled artifact for execution.
///
/// The single production entry point between compilation and sessions. It runs
/// graph-driven weight materialization, builds the execution plan against
/// `runtime`'s backends, and derives the immutable weight/constant binding table
/// from the same requirement query that PrepareExecutionBindings validates
/// against, so the two can never disagree. Completeness is reconciled here: a
/// weight that does not resolve, or a constant with no inline data, fails
/// preparation instead of surfacing as wrong numerics at execution time.
///
/// @param runtime Runtime providing the backends the plan resolves kernels
///        through; must outlive the returned model.
/// @param artifact Compiled artifact to consume. Ownership transfers into the
///        returned model, which keeps it alive as the weight backing.
/// @return The prepared model, or an error if any weight or constant cannot be
///         materialized, the artifact mixes execution phases, or plan building
///         fails. No partially prepared model is observable on failure.
AM_NODISCARD StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact);

} // namespace aethermind

#endif
