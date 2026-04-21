#include "aethermind/execution/runtime_binding_context.h"

namespace aethermind {

void RuntimeBindingContext::SetWorkspaceArena(WorkspaceArena* workspace_arena) noexcept {
    workspace_arena_ = workspace_arena;
}

WorkspaceArena* RuntimeBindingContext::GetWorkspaceArena() const noexcept {
    return workspace_arena_;
}

StatusOr<WorkspaceBinding> RuntimeBindingContext::BindWorkspace(
        const WorkspaceRequirement& requirement) const noexcept {
    if (!IsValidWorkspaceAlignment(requirement.alignment)) {
        return Status::InvalidArgument(
                "Workspace requirement alignment must be a non-zero power of two");
    }

    if (requirement.bytes == 0) {
        return WorkspaceBinding{};
    }

    if (workspace_arena_ == nullptr) {
        return Status(StatusCode::kFailedPrecondition,
                      "RuntimeBindingContext requires a WorkspaceArena for non-zero workspace requirements");
    }

    const auto binding = workspace_arena_->Bind(requirement);
    if (binding.data == nullptr || binding.size < requirement.bytes) {
        return Status(StatusCode::kFailedPrecondition,
                      "WorkspaceArena could not satisfy execution-time workspace binding");
    }
    return binding;
}

void RuntimeBindingContext::SetKVCacheView(const KVCacheView& kv_cache_view) noexcept {
    kv_cache_view_ = kv_cache_view;
}

const KVCacheView& RuntimeBindingContext::kv_cache_view() const noexcept {
    return kv_cache_view_;
}

bool RuntimeBindingContext::HasKVCacheView() const noexcept {
    return kv_cache_view_.valid();
}

void RuntimeBindingContext::ClearKVCacheView() noexcept {
    kv_cache_view_ = {};
}

void RuntimeBindingContext::SetTempBufferBinding(
        TempBufferKind kind,
        const TempBufferBinding& binding) noexcept {
    temp_buffers_[ToIndex(kind)] = binding;
}

const TempBufferBinding& RuntimeBindingContext::GetTempBufferBinding(
        TempBufferKind kind) const noexcept {
    return temp_buffers_[ToIndex(kind)];
}

bool RuntimeBindingContext::HasTempBufferBinding(TempBufferKind kind) const noexcept {
    return GetTempBufferBinding(kind).valid();
}

void RuntimeBindingContext::ClearTempBufferBinding(TempBufferKind kind) noexcept {
    temp_buffers_[ToIndex(kind)] = {};
}

void RuntimeBindingContext::ClearAllTempBufferBindings() noexcept {
    temp_buffers_.fill({});
}

RuntimeSequenceState& RuntimeBindingContext::mutable_sequence_state() noexcept {
    return sequence_state_;
}

const RuntimeSequenceState& RuntimeBindingContext::sequence_state() const noexcept {
    return sequence_state_;
}

void RuntimeBindingContext::ResetSequenceState() noexcept {
    sequence_state_ = {};
}

void RuntimeBindingContext::Reset() noexcept {
    if (workspace_arena_ != nullptr) {
        workspace_arena_->Reset();
    }
    ClearKVCacheView();
    ClearAllTempBufferBindings();
    ResetSequenceState();
}

}// namespace aethermind
