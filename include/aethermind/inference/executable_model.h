#ifndef AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H
#define AETHERMIND_INFERENCE_EXECUTABLE_MODEL_H

/// @file executable_model.h
/// @brief Production-ready model prepared from a compiler artifact.

#include "aethermind/base/kernel_attrs.h"
#include "aethermind/base/status.h"
#include "aethermind/compiler/model_compiler.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/inference/inference_options.h"
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
struct ModelLimits {
    size_t max_model_payload_bytes = 0;
    size_t max_packed_weight_bytes = 0;
};

/// @brief Explicit source-weight admission and conversion policy.
enum class WeightPolicy : uint8_t {
    kRequireFp32,
    kConvertToFp32,
};

/// @brief Target, compilation and source-weight policy with per-model caps.
///
/// Compatibility widens actual F16/BF16 weights once before compilation and
/// runs FP32 weights, activations, KV and logits. It requires CPU O1/plain.
/// The strict policy keeps the existing FP32-only source admission contract.
struct ModelLoadOptions {
    CompilePreset preset = CompilePreset::kO1Plain;
    DeviceType target = DeviceType::kCPU;
    WeightPolicy weights = WeightPolicy::kRequireFp32;
    ModelLimits limits{};
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
/// - the Runtime used to prepare this model must remain at the same
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
    /// @brief Loads a model for a target using an explicit weight policy and preset.
    /// @note Loads once; conversion precedes graph construction. Runtime must
    ///       remain at the same address and outlive the returned model. CLI and
    ///       C ABI retain their CPU/strict source admission through staged calls.
    ///       Preparation checks the target backend and kernel availability.
    static StatusOr<ExecutableModel> Load(Runtime& runtime,
                                          const std::filesystem::path& directory,
                                          const ModelLoadOptions& options);

    /// @brief Prepares a compiled artifact, consuming its weight backing.
    /// @note Checks payload limits before packing, resolves kernels and builds
    ///       immutable bindings. Failure publishes no partial model. This never
    ///       converts source weights; Runtime must remain stable and outlive it.
    static StatusOr<ExecutableModel> Prepare(Runtime& runtime,
                                             LoweredModelArtifact artifact,
                                             const ModelLimits& limits = {});

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

} // namespace aethermind

#endif
