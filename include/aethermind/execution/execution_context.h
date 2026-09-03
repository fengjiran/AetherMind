#ifndef AETHERMIND_EXECUTION_EXECUTION_CONTEXT_H
#define AETHERMIND_EXECUTION_EXECUTION_CONTEXT_H

/// @file execution_context.h
/// @brief Narrow execution-time resources for one ExecutionPlan specialization.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/base/workspace_arena.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/runtime/kv_cache_view.h"

namespace aethermind {

/// @brief Aggregates the resources needed to execute one specialized plan.
///
/// The context owns PreparedExecutionBindings and borrows a WorkspaceArena.
/// KVCacheView is a non-owning view of Runtime-owned KV storage. Use Create()
/// before Execute() so plan identity, required workspace, and state-alias KV
/// presence are checked on the cold path.
class ExecutionContext {
public:
    /// @brief Creates an executable context for one plan specialization.
    ///
    /// @param plan Immutable plan that will be executed. The context does not
    ///             retain it, so it must outlive every Execute() using the
    ///             returned context.
    /// @param prepared_bindings Cold-path bindings prepared for `plan`.
    /// @param workspace_arena Borrowed workspace arena; required when `plan`
    ///                        has non-zero workspace requirements.
    /// @param kv_cache_view Borrowed KV storage view; required when `plan`
    ///                      contains state aliases.
    /// @return A complete context, or an error if its resources cannot serve
    ///         the plan. Workspace capacity and KV geometry are checked when
    ///         their individual steps bind them.
    static StatusOr<ExecutionContext> Create(
            const ExecutionPlan& plan,
            PreparedExecutionBindings prepared_bindings,
            WorkspaceArena* workspace_arena,
            KVCacheView kv_cache_view = {});

    ExecutionContext(const ExecutionContext&) = delete;
    ExecutionContext& operator=(const ExecutionContext&) = delete;
    ExecutionContext(ExecutionContext&&) noexcept = default;
    ExecutionContext& operator=(ExecutionContext&&) noexcept = default;
    ~ExecutionContext() = default;

    /// @brief Returns the borrowed workspace arena, or nullptr.
    AM_NODISCARD WorkspaceArena* workspace_arena() const noexcept;

    /// @brief Binds a workspace allocation for a step requirement.
    ///
    /// @param requirement Required bytes and alignment.
    /// @return A binding of at least the required size, or an error if no
    ///         arena is bound or the arena cannot satisfy the requirement.
    AM_NODISCARD StatusOr<WorkspaceBinding> BindWorkspace(
            const WorkspaceRequirement& requirement) const noexcept;

    /// @brief Returns the current borrowed KV cache view.
    AM_NODISCARD const KVCacheView& kv_cache_view() const noexcept;

    /// @brief Returns whether a valid KV cache view is present.
    AM_NODISCARD bool HasKVCacheView() const noexcept;

    /// @brief Returns prepared tensor and kernel bindings, or nullptr when
    /// the context has been cleared or was not created for execution.
    AM_NODISCARD const PreparedExecutionBindings* prepared_bindings() const noexcept;

    /// @brief Clears owned bindings and borrowed handles.
    ///
    /// This does not reset the borrowed WorkspaceArena or release the KV
    /// reservation; those resources are owned by the caller/session.
    void Clear() noexcept;

private:
    explicit ExecutionContext(WorkspaceArena* workspace_arena) noexcept
        : workspace_arena_(workspace_arena) {}

    WorkspaceArena* workspace_arena_ = nullptr;
    KVCacheView kv_cache_view_{};
    PreparedExecutionBindings prepared_bindings_{};
};

} // namespace aethermind

#endif
