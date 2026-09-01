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
/// runtime state remain owned by RuntimeBindingContext.
struct ExternalValueBindings {
    std::vector<ExternalReadOnlyValueBinding> readable{};
    std::vector<ExternalWritableValueBinding> writable{};
};

/// @brief Canonical physical binding for one ExecutionValueId.
///
/// The views borrow either caller-owned storage or BindingTable-owned
/// activation storage and concrete metadata. They remain valid for the table
/// lifetime.
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

class BindingTableStorage;

/// @brief Controlled result of specializing one ExecutionPlan for a call/session.
///
/// The type owns activation storage and concrete metadata. Its heap-stable
/// storage keeps cached TensorView metadata valid across moves.
class BindingTable {
public:
    BindingTable() noexcept;
    BindingTable(BindingTable&&) noexcept;
    BindingTable& operator=(BindingTable&&) noexcept;
    BindingTable(const BindingTable&) = delete;
    BindingTable& operator=(const BindingTable&) = delete;
    ~BindingTable();

    AM_NODISCARD std::span<const BoundValue> values() const noexcept;
    AM_NODISCARD const StepTensorBinding& step(size_t step_index) const noexcept;
    AM_NODISCARD size_t step_count() const noexcept;
    AM_NODISCARD bool empty() const noexcept;
    AM_NODISCARD bool IsCompatible(const ExecutionPlan& plan) const noexcept;

private:
    friend StatusOr<BindingTable> BuildExecutionBindings(
            const ExecutionPlan& plan,
            const ExternalValueBindings& external,
            Allocator& act_allocator);

    explicit BindingTable(std::unique_ptr<BindingTableStorage> storage) noexcept;

    std::unique_ptr<BindingTableStorage> storage_{};
};

/// @brief Resolves external tensors and allocates internal activations.
///
/// This is a cold-path operation. It validates external dtypes/ranks/static
/// dimensions and symbolic identity, assigns each logical value exactly one
/// canonical storage location, allocates the activation arena, and caches all
/// compact per-step TensorViews. The allocator must outlive this call; the
/// returned BindingTable owns its resulting Buffer.
StatusOr<BindingTable> BuildExecutionBindings(
        const ExecutionPlan& plan,
        const ExternalValueBindings& external,
        Allocator& act_allocator);

} // namespace aethermind

#endif
