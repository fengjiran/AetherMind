#include "aethermind/execution/execution_context.h"

#include <algorithm>

namespace aethermind {

StatusOr<ExecutionContext> ExecutionContext::Create(
        const ExecutionPlan& plan,
        PreparedExecutionBindings prepared_bindings,
        WorkspaceArena* workspace_arena,
        KVCacheView kv_cache_view) {
    if (!prepared_bindings.IsCompatible(plan)) {
        return Status::InvalidArgument(
                "PreparedExecutionBindings are not compatible with ExecutionPlan");
    }

    if (plan.total_workspace_bytes() != 0 && workspace_arena == nullptr) {
        return Status::FailedPrecondition(
                "ExecutionPlan requires a WorkspaceArena for non-zero workspace requirements");
    }

    const bool requires_kv_cache = std::ranges::any_of(
            plan.values(), [](const ExecutionValueDesc& value) {
                return value.kind == ExecutionValueKind::kState;
            });
    if (requires_kv_cache && !kv_cache_view.valid()) {
        return Status::FailedPrecondition(
                "ExecutionPlan state values require a valid KVCacheView");
    }

    ExecutionContext context(workspace_arena);
    context.kv_cache_view_ = kv_cache_view;
    if (requires_kv_cache) {
        context.kv_layer_transaction_scratch_.resize(kv_cache_view.num_layers());
    }
    context.prepared_bindings_ = std::move(prepared_bindings);
    return context;
}

WorkspaceArena* ExecutionContext::workspace_arena() const noexcept {
    return workspace_arena_;
}

StatusOr<WorkspaceBinding> ExecutionContext::BindWorkspace(
        const WorkspaceRequirement& requirement) const noexcept {
    if (!IsValidWorkspaceAlignment(requirement.alignment)) {
        return Status::InvalidArgument(
                "Workspace requirement alignment must be a non-zero power of two");
    }

    if (requirement.bytes == 0) {
        return WorkspaceBinding{};
    }

    if (workspace_arena_ == nullptr) {
        return Status::FailedPrecondition(
                "ExecutionContext requires a WorkspaceArena for non-zero workspace requirements");
    }

    const WorkspaceBinding binding = workspace_arena_->Bind(requirement);
    if (binding.data == nullptr || binding.size < requirement.bytes) {
        return Status::FailedPrecondition(
                "WorkspaceArena could not satisfy execution-time workspace binding");
    }
    return binding;
}

const KVCacheView& ExecutionContext::kv_cache_view() const noexcept {
    return kv_cache_view_;
}

bool ExecutionContext::HasKVCacheView() const noexcept {
    return kv_cache_view_.valid();
}

const PreparedExecutionBindings* ExecutionContext::prepared_bindings() const noexcept {
    return prepared_bindings_.empty() ? nullptr : &prepared_bindings_;
}

std::span<uint8_t> ExecutionContext::ResetKVLayerTransactionScratch() noexcept {
    std::ranges::fill(kv_layer_transaction_scratch_, uint8_t{0});
    return kv_layer_transaction_scratch_;
}

void ExecutionContext::Clear() noexcept {
    prepared_bindings_ = {};
    kv_cache_view_ = {};
    kv_layer_transaction_scratch_.clear();
    workspace_arena_ = nullptr;
}

} // namespace aethermind
