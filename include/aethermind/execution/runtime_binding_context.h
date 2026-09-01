#ifndef AETHERMIND_EXECUTION_RUNTIME_BINDING_CONTEXT_H
#define AETHERMIND_EXECUTION_RUNTIME_BINDING_CONTEXT_H

/// @file runtime_binding_context.h
/// @brief Per-call runtime inputs and bindings for plan execution.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/base/workspace_arena.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/runtime/kv_cache_view.h"

#include <array>

namespace aethermind {

/// @brief Roles of caller-owned temporary buffers.
enum class TempBufferKind : size_t {
    kHiddenState = 0,
    kLogits = 1,
    kScratch = 2,
    kCount,
};

/// @brief Borrowed temporary buffer with its byte size.
///
/// A binding is valid when both `data` and `size` are non-zero.
struct TempBufferBinding {
    void* data = nullptr;
    size_t size = 0;

    /// @brief Returns whether the binding points at a non-empty buffer.
    AM_NODISCARD bool valid() const noexcept {
        return data != nullptr && size != 0;
    }
};

/// @brief Progress of the current inference sequence.
struct RuntimeSequenceState {
    bool prepared = false;
    bool prefill_done = false;
    size_t prompt_len = 0;
    size_t generated_len = 0;
    size_t current_pos = 0;
};

/// @brief Aggregates the runtime inputs a plan execution needs.
///
/// Owned by the caller and populated before Execute(). Holds a borrowed
/// WorkspaceArena (must outlive execution), an optional KV cache view,
/// caller-owned temporary buffers, sequence progress state, and a controlled
/// BindingTable that caches per-value and per-step tensor bindings.
class RuntimeBindingContext {
public:
    RuntimeBindingContext() = default;
    /// @brief Binds the workspace arena used for execution-time allocations.
    ///
    /// @param workspace_arena Borrowed arena; must outlive execution.
    explicit RuntimeBindingContext(WorkspaceArena* workspace_arena) noexcept
        : workspace_arena_(workspace_arena) {}

    /// @brief Binds the workspace arena used for execution-time allocations.
    ///
    /// @param workspace_arena Borrowed arena; must outlive execution.
    void SetWorkspaceArena(WorkspaceArena* workspace_arena) noexcept;

    /// @brief Returns the bound workspace arena, or nullptr.
    AM_NODISCARD WorkspaceArena* GetWorkspaceArena() const noexcept;

    /// @brief Binds a workspace allocation for a step requirement.
    ///
    /// @param requirement Required bytes and alignment.
    /// @return A binding of at least the required size, or an error if no
    ///         arena is bound or the arena cannot satisfy the requirement.
    AM_NODISCARD StatusOr<WorkspaceBinding> BindWorkspace(
            const WorkspaceRequirement& requirement) const noexcept;

    /// @brief Sets the KV cache view used by state-alias steps.
    ///
    /// @param kv_cache_view Copied into the context.
    void SetKVCacheView(const KVCacheView& kv_cache_view) noexcept;

    /// @brief Returns the current KV cache view.
    AM_NODISCARD const KVCacheView& kv_cache_view() const noexcept;

    /// @brief Returns whether a valid KV cache view is set.
    AM_NODISCARD bool HasKVCacheView() const noexcept;

    /// @brief Clears the KV cache view.
    void ClearKVCacheView() noexcept;

    /// @brief Sets a caller-owned temporary buffer for a role.
    ///
    /// @param kind Buffer role.
    /// @param binding Borrowed buffer; must outlive execution.
    void SetTempBufferBinding(TempBufferKind kind,
                              const TempBufferBinding& binding) noexcept;

    /// @brief Returns the temporary buffer bound to a role.
    ///
    /// @param kind Buffer role.
    /// @return The stored binding (possibly invalid).
    AM_NODISCARD const TempBufferBinding& GetTempBufferBinding(TempBufferKind kind) const noexcept;

    /// @brief Returns whether a role has a valid temporary buffer bound.
    ///
    /// @param kind Buffer role.
    /// @return True if a non-empty binding is stored.
    AM_NODISCARD bool HasTempBufferBinding(TempBufferKind kind) const noexcept;

    /// @brief Clears a single temporary buffer role.
    ///
    /// @param kind Buffer role to clear.
    void ClearTempBufferBinding(TempBufferKind kind) noexcept;

    /// @brief Clears all temporary buffer roles.
    void ClearAllTempBufferBindings() noexcept;

    /// @brief Returns mutable access to the sequence progress state.
    AM_NODISCARD RuntimeSequenceState& mutable_sequence_state() noexcept;

    /// @brief Returns the sequence progress state.
    AM_NODISCARD const RuntimeSequenceState& sequence_state() const noexcept;

    /// @brief Resets the sequence progress state.
    void ResetSequenceState() noexcept;

    /// @brief Installs the cold-path binding cache for the current plan/session.
    void SetBindingTable(BindingTable binding_table) noexcept;

    /// @brief Returns the binding cache, or nullptr when none is installed.
    AM_NODISCARD const BindingTable* binding_table() const noexcept;

    /// @brief Resets workspace arena, KV cache view, temporary buffers,
    ///        the binding table, and sequence state to defaults.
    void Reset() noexcept;

private:
    static constexpr size_t ToIndex(TempBufferKind kind) noexcept {
        return static_cast<size_t>(kind);
    }

    WorkspaceArena* workspace_arena_ = nullptr;
    KVCacheView kv_cache_view_{};
    std::array<TempBufferBinding, static_cast<size_t>(TempBufferKind::kCount)> temp_buffers_{};
    RuntimeSequenceState sequence_state_{};
    BindingTable binding_table_{};
};

} // namespace aethermind

#endif
