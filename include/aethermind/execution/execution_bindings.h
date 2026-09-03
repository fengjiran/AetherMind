#ifndef AETHERMIND_EXECUTION_EXECUTION_BINDINGS_H
#define AETHERMIND_EXECUTION_EXECUTION_BINDINGS_H

/// @file execution_bindings.h
/// @brief Cold-path value binding and cached kernel operands for an ExecutionPlan.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"

#include <memory>
#include <span>
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
    std::vector<ExternalReadOnlyValueBinding> readable{};
    std::vector<ExternalWritableValueBinding> writable{};
};

/// @brief Canonical physical binding for one ExecutionValueId.
///
/// The views borrow either caller-owned storage or PreparedExecutionBindings-owned
/// activation storage. Concrete shape and stride metadata are owned by the
/// prepared bindings, but caller-owned tensor storage must outlive them.
struct BoundValue {
    TensorView readable{};
    MutableTensorView writable{};
    bool has_writable = false;
};

/// @brief Per-step compact TensorViews cached for the executor hot path.
struct StepTensorBinding {
    std::vector<TensorView> inputs{};
    std::vector<MutableTensorView> outputs{};
};

class PreparedExecutionBindingsStorage;

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
    friend StatusOr<PreparedExecutionBindings> PrepareExecutionBindings(
            const ExecutionPlan& plan,
            const ExternalTensorBindings& external,
            Allocator& act_allocator);

    explicit PreparedExecutionBindings(
            std::unique_ptr<PreparedExecutionBindingsStorage> storage) noexcept;

    std::unique_ptr<PreparedExecutionBindingsStorage> storage_{};
};

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
