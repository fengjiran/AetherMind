#ifndef AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H
#define AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H

/// @file executable_model.h
/// @brief Production-ready model prepared from a compiler artifact.

#include "aethermind/base/kernel_attrs.h"
#include "aethermind/base/status.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/inference/resource_usage.h"
#include "aethermind/inference/weight_binding_metadata.h"
#include "aethermind/model/weight/packed_weight_collection.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>

namespace aethermind {

class Runtime;

/// @brief Per-model visible tensor payload caps; zero means unlimited.
/// Mapped/external backing, metadata and allocator overhead are reported or
/// excluded separately. A model cap requires known raw backing metadata.
struct ExecutableModelPrepareOptions {
    size_t max_model_payload_bytes = 0;
    size_t max_packed_weight_bytes = 0;
};

/// @brief Explicit source-weight policy for the CPU preparation entry point.
enum class InferencePrecisionPreset : uint8_t {
    kStrictFp32,
    kFp32Compatible,
};

/// @brief CPU execution policy and per-model payload caps.
///
/// Compatibility widens actual F16/BF16 weights once before compilation and
/// runs FP32 weights, activations, KV and logits. It currently requires O1/plain.
/// The strict policy keeps the existing FP32-only source admission contract.
struct CpuModelPrepareOptions {
    CpuInferencePreset cpu_preset = CpuInferencePreset::kO1Plain;
    InferencePrecisionPreset precision = InferencePrecisionPreset::kStrictFp32;
    ExecutableModelPrepareOptions resources{};
};

/// @brief Immutable, execution-ready model owned between compilation and sessions.
///
/// Owns everything a session needs to specialize and run the model: the compiler
/// artifact and through it the raw weight backing, the packed-weight artifacts,
/// the metadata that the immutable binding table borrows, that table, and the
/// execution plan. Sessions never read the compiler artifact, resolve weights by
/// role, or interpret packing decisions — preparation has already done it.
///
/// Lifetime, outermost first:
/// - the Runtime passed to PrepareExecutableModel must remain at the same
///   address, must not be moved, and must outlive this object,
///   because the plan's resolved kernels borrow its backends;
/// - this object must outlive every PreparedExecutionBindings built from its plan
///   and bindings, because prepared bindings borrow the weight data pointers
///   recorded here;
/// - member declaration order is the teardown contract: destruction runs in
///   reverse, releasing the plan and the binding table before the metadata and
///   the artifact backing they borrow.
///
/// Move-only: movable but not assignable. Assignment is deleted because
/// replacing a prepared model in place would invalidate every plan and binding
/// pointer already handed out, and because member-wise assignment would release
/// the artifact backing before the binding table that borrows it. Replace a
/// model by constructing a new one.
///
/// Not thread-safe: preparation is a cold-path operation and callers
/// must serialize concurrent access.
class ExecutableModel {
public:
    ExecutableModel(ExecutableModel&&) noexcept = default;
    ExecutableModel& operator=(ExecutableModel&&) = delete;
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
    ///        from, matching the packed-weight collection's source id.
    AM_NODISCARD uint64_t artifact_id() const noexcept;

    /// @brief Returns the single phase this artifact was compiled for.
    AM_NODISCARD ExecPhase phase() const noexcept;

    /// @brief Returns the conservative maximum sequence length from the
    ///        validated HF max_position_embeddings field.
    AM_NODISCARD size_t context_limit() const noexcept;

    /// @brief Returns the validated vocabulary size used to check token IDs.
    AM_NODISCARD size_t vocab_size() const noexcept;

    /// @brief Returns whether this model's resolved kernels belong to runtime.
    AM_NODISCARD bool IsPreparedFor(const Runtime& runtime) const noexcept;

    /// @brief Returns immutable preparation accounting for this model only.
    AM_NODISCARD const ModelResourceUsage& resource_usage() const noexcept;

private:
    friend StatusOr<ExecutableModel> PrepareExecutableModel(Runtime& runtime,
                                                            LoweredModelArtifact artifact,
                                                            const ExecutableModelPrepareOptions& options);

    ExecutableModel(Runtime& runtime,
                    LoweredModelArtifact artifact,
                    PackedWeightCollection packed_weights,
                    WeightBindingMetadata binding_metadata,
                    ExternalTensorBindings bindings,
                    ExecutionPlan plan,
                    ExecPhase phase,
                    size_t context_limit,
                    size_t vocab_size,
                    ModelResourceUsage resource_usage) noexcept;

    /// @brief Validates a phase query against the compiled artifact phase.
    Status CheckPhase(ExecPhase phase) const noexcept;

    // The resolved kernel function pointers borrow backend-owned state.
    // This Runtime must stay at the same address and outlive this model.
    Runtime* runtime_ = nullptr;
    LoweredModelArtifact artifact_{};
    PackedWeightCollection packed_weights_{};
    WeightBindingMetadata binding_metadata_{};
    ExternalTensorBindings bindings_{};
    ExecutionPlan plan_{};
    ExecPhase phase_ = ExecPhase::kBoth;
    size_t context_limit_ = 0;
    size_t vocab_size_ = 0;
    ModelResourceUsage resource_usage_{};
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
StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact);

/// @brief Prepares with per-model payload admission; no partial model is published.
StatusOr<ExecutableModel> PrepareExecutableModel(
        Runtime& runtime,
        LoweredModelArtifact artifact,
        const ExecutableModelPrepareOptions& options);

/// @brief Loads, compiles, and prepares a model directory for execution.
///
/// This is a convenience orchestration entry point over
/// `ModelCompiler::LoadAndCompile` and `PrepareExecutableModel`. Compilation
/// options are explicit because the default O2/plain-weight combination does
/// not currently resolve against the CPU kernel registry. The returned model
/// borrows the backend owned by `runtime`; `runtime` must remain at the same address and
/// outlive the returned model.
///
/// @param runtime Runtime providing the backends used to resolve kernels.
/// @param model_dir Directory containing the model configuration and weights.
/// @param compile_options Semantic optimization and lowering configuration.
/// @return The prepared model, or an error with the original status code and
///         context identifying loading/compilation or preparation.
StatusOr<ExecutableModel> LoadAndPrepareExecutableModel(
        Runtime& runtime,
        const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options);

/// @brief Uses the same load/compile/prepare chain with explicit payload caps.
StatusOr<ExecutableModel> LoadAndPrepareExecutableModel(
        Runtime& runtime,
        const std::filesystem::path& model_dir,
        const ModelCompileOptions& compile_options,
        const ExecutableModelPrepareOptions& prepare_options);

/// @brief Loads once and prepares an explicitly selected CPU precision policy.
///
/// The Runtime must already be at its final address and outlive the model.
/// Compatibility uses its CPU allocator for owned FP32 conversion storage,
/// releases unused source backing before compilation, and preserves the original
/// config dtype hint as provenance. The model cap is also checked against the
/// normalized owned weights before allocating conversion payloads. It does not
/// bound source mappings, conversion-time RSS, or allocator overhead.
///
/// Low-level compiler/preparation overloads retain their existing behavior;
/// they never normalize weights implicitly. The C ABI and CLI retain strict
/// source admission.
StatusOr<ExecutableModel> LoadAndPrepareCpuExecutableModel(
        Runtime& runtime,
        const std::filesystem::path& model_dir,
        const CpuModelPrepareOptions& options);

} // namespace aethermind

#endif
