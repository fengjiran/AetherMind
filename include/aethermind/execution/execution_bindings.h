#ifndef AETHERMIND_EXECUTION_EXECUTION_BINDINGS_H
#define AETHERMIND_EXECUTION_EXECUTION_BINDINGS_H

/// @file execution_bindings.h
/// @brief Cold-path value binding and cached kernel operands for an ExecutionPlan.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"

#include <memory>
#include <span>
#include <variant>
#include <vector>

namespace aethermind {

class Allocator;

/// @brief Read-only external tensor bound to one logical execution value.
struct ExternalReadOnlyValueBinding {
    ExecutionValueId value{};
    TensorView tensor{};
};

/// @brief Writable external tensor bound to one logical execution value.
struct ExternalWritableValueBinding {
    ExecutionValueId value{};
    MutableTensorView tensor{};
};

/// @brief Caller-provided tensors needed to specialize an ExecutionPlan.
///
/// State values intentionally are not represented here: KV and other opaque
/// runtime state remain owned by ExecutionContext.
struct ExternalTensorBindings {
    std::vector<ExternalReadOnlyValueBinding> read_only{};
    std::vector<ExternalWritableValueBinding> writable{};
};

/// @brief Canonical physical binding for one ExecutionValueId.
///
/// The view borrows either caller-owned storage or PreparedExecutionBindings-owned
/// activation storage. Concrete shape and stride metadata are owned by the
/// prepared bindings, but caller-owned tensor storage must outlive them.
/// An unbound value has no dense tensor view, as for state values or weights
/// consumed only through packed artifacts. A writable binding also permits reads;
/// its read-only descriptor is derived from the same view rather than stored twice.
class BoundValue {
public:
    /// @brief Creates an unbound value.
    BoundValue() noexcept = default;

    /// @brief Creates a read-only binding, or an unbound value for an invalid view.
    /// @param view Borrowed tensor data and metadata, which must outlive the binding.
    explicit BoundValue(TensorView view) noexcept;

    /// @brief Creates a writable binding, or an unbound value for an invalid view.
    /// @param view Borrowed tensor data and metadata, which must outlive the binding.
    explicit BoundValue(MutableTensorView view) noexcept;

    /// @brief Returns a read-only descriptor without copying the tensor data.
    /// @return A borrowed view of either binding kind, or an invalid view when unbound.
    /// @note The returned descriptor borrows data and metadata, not this object;
    ///       that backing storage must remain valid for the descriptor's lifetime.
    AM_NODISCARD TensorView ReadView() const noexcept;

    /// @brief Returns the writable descriptor when this binding permits writes.
    /// @return A pointer into this object, or nullptr for unbound/read-only values.
    ///         The pointer must not outlive this object's modification, move, or destruction.
    AM_NODISCARD const MutableTensorView* TryGetWritableView() const noexcept;

private:
    std::variant<std::monostate, TensorView, MutableTensorView> view_{};
};

/// @brief Per-step compact TensorViews cached for the executor hot path.
struct StepTensorBinding {
    std::vector<TensorView> inputs{};
    std::vector<MutableTensorView> outputs{};
};

class PreparedExecutionBindingsStorage;
class ExecutionBindingPlan;

/// @brief Controlled result of specializing one ExecutionPlan for a call/session.
///
/// The type owns activation storage and concrete metadata. Its heap-stable
/// storage keeps cached TensorView metadata valid across moves. It borrows the
/// external tensor data pointers supplied during preparation, so those backing
/// allocations must remain valid and unchanged until the bindings are rebuilt.
class PreparedExecutionBindings {
public:
    PreparedExecutionBindings() noexcept;
    PreparedExecutionBindings(PreparedExecutionBindings&&) noexcept;
    PreparedExecutionBindings& operator=(PreparedExecutionBindings&&) noexcept;
    PreparedExecutionBindings(const PreparedExecutionBindings&) = delete;
    PreparedExecutionBindings& operator=(const PreparedExecutionBindings&) = delete;
    ~PreparedExecutionBindings();

    AM_NODISCARD std::span<const BoundValue> values() const noexcept;
    AM_NODISCARD const StepTensorBinding& step(size_t step_index) const noexcept;

    /// @brief Prepared kernel params for one step, or nullptr when the step's
    /// kernel registered no params builder.
    ///
    /// The pointer is owned by these bindings and stays valid across
    /// PreparedExecutionBindings moves. Content is immutable for the bindings'
    /// lifetime; per-execution state is never baked into it.
    AM_NODISCARD const void* kernel_params(size_t step_index) const noexcept;

    AM_NODISCARD size_t step_count() const noexcept;
    AM_NODISCARD bool empty() const noexcept;
    AM_NODISCARD bool IsCompatible(const ExecutionPlan& plan) const noexcept;

private:
    friend StatusOr<PreparedExecutionBindings> MaterializeExecutionBindings(
            const ExecutionPlan& plan,
            ExecutionBindingPlan binding_plan,
            Allocator& act_allocator);
    friend StatusOr<PreparedExecutionBindings> PrepareExecutionBindings(
            const ExecutionPlan& plan,
            const ExternalTensorBindings& external,
            Allocator& act_allocator);

    explicit PreparedExecutionBindings(
            std::unique_ptr<PreparedExecutionBindingsStorage> storage) noexcept;

    std::unique_ptr<PreparedExecutionBindingsStorage> storage_{};
};

/// @brief Controlled activation layout and external metadata for one concrete call.
///
/// Planning owns concrete shape/stride snapshots and borrows external data
/// addresses, without allocating activation storage or building kernel params.
/// External backing must outlive materialization and the resulting bindings;
/// its address/layout must remain stable, while input payload values may change.
/// The plan's heap-stable metadata remains valid across moves. Empty and
/// moved-from objects cannot be materialized.
class ExecutionBindingPlan {
public:
    ExecutionBindingPlan() noexcept;
    ExecutionBindingPlan(ExecutionBindingPlan&&) noexcept;
    ExecutionBindingPlan& operator=(ExecutionBindingPlan&&) noexcept;
    ExecutionBindingPlan(const ExecutionBindingPlan&) = delete;
    ExecutionBindingPlan& operator=(const ExecutionBindingPlan&) = delete;
    ~ExecutionBindingPlan();

    /// @brief Returns the exact visible bytes of the planned activation arena.
    AM_NODISCARD size_t activation_bytes() const noexcept;
    /// @brief Returns the arena's required alignment, or zero when empty.
    AM_NODISCARD size_t activation_alignment() const noexcept;
    /// @brief Checks the immutable plan identity and planned value/step counts.
    AM_NODISCARD bool IsCompatible(const ExecutionPlan& plan) const noexcept;

private:
    friend StatusOr<ExecutionBindingPlan> PlanExecutionBindings(
            const ExecutionPlan& plan, const ExternalTensorBindings& external);

    friend StatusOr<PreparedExecutionBindings> MaterializeExecutionBindings(
            const ExecutionPlan& plan,
            ExecutionBindingPlan binding_plan,
            Allocator& act_allocator);

    ExecutionBindingPlan(std::unique_ptr<PreparedExecutionBindingsStorage> storage,
                         std::vector<size_t> activation_offsets,
                         size_t activation_bytes,
                         size_t step_count) noexcept;

    std::unique_ptr<PreparedExecutionBindingsStorage> storage_{};
    std::vector<size_t> activation_offsets_{};
    size_t activation_bytes_ = 0;
    size_t step_count_ = 0;
};

/// @brief Reports which plan values need a caller-supplied read-only binding.
///
/// The result is indexed by ExecutionValueId::index and aligned with
/// ExecutionPlan::values(). Model inputs and constants are always required. A
/// weight value is required only while some step still consumes it through a
/// kernel input port: a step selecting WeightFormat::kPacked has its kWeight
/// ports projected out, so such weights are served by the plan's packed
/// artifacts and must not be bound externally. Activations and state values are
/// never required — PrepareExecutionBindings allocates activations, and state
/// stays owned by ExecutionContext.
///
/// PrepareExecutionBindings consults this same function, so supplying exactly
/// the reported set satisfies its read-only completeness check by construction.
/// Callers that materialize weight bindings should drive from this query rather
/// than re-deriving the plain/packed decision.
///
/// @param plan Validated plan to inspect.
/// @return Per-value requirement flags, or an error when the plan's kernel port
///         metadata is inconsistent with its semantic inputs.
StatusOr<std::vector<bool>> ComputeExternalReadRequirements(
        const ExecutionPlan& plan);

/// @brief Validates external bindings and plans concrete activation storage.
///
/// Snapshots shape/stride metadata, resolves symbols, and assigns arena offsets
/// using the same layout consumed by MaterializeExecutionBindings. Reads no
/// tensor payload and creates no activation Buffer or prepared kernel params.
/// Metadata allocation may throw std::bad_alloc, as PrepareExecutionBindings does.
StatusOr<ExecutionBindingPlan> PlanExecutionBindings(
        const ExecutionPlan& plan,
        const ExternalTensorBindings& external);

/// @brief Allocates and binds the exact arena described by a controlled plan.
///
/// Consumes binding_plan and rejects a different, empty, or moved-from execution
/// plan before invoking the allocator. Preserves the normal tensor/alias,
/// deferred-shape and params-builder validation before bindings are published.
/// Allocator borrowing and external storage lifetime match PrepareExecutionBindings.
StatusOr<PreparedExecutionBindings> MaterializeExecutionBindings(
        const ExecutionPlan& plan,
        ExecutionBindingPlan binding_plan,
        Allocator& act_allocator);

/// @brief Resolves external tensors and allocates internal activations.
///
/// This is a cold-path operation. It validates external dtypes/ranks/static
/// dimensions and symbolic identity, assigns each logical value exactly one
/// canonical storage location, allocates the activation arena, and caches all
/// compact per-step TensorViews. The allocator must outlive this call; the
/// returned PreparedExecutionBindings owns its resulting Buffer. External
/// tensor backing storage is borrowed and must outlive the returned bindings.
StatusOr<PreparedExecutionBindings> PrepareExecutionBindings(
        const ExecutionPlan& plan,
        const ExternalTensorBindings& external,
        Allocator& act_allocator);

} // namespace aethermind

#endif
