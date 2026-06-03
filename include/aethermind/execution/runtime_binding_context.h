#ifndef AETHERMIND_EXECUTION_RUNTIME_BINDING_CONTEXT_H
#define AETHERMIND_EXECUTION_RUNTIME_BINDING_CONTEXT_H

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/kv_cache_view.h"
#include "aethermind/runtime/workspace.h"
#include "macros.h"
#include "workspace_arena.h"

#include <array>
#include <vector>

namespace aethermind {

enum class TempBufferKind : size_t {
    kHiddenState = 0,
    kLogits = 1,
    kScratch = 2,
    kCount,
};

struct TempBufferBinding {
    void* data = nullptr;
    size_t size = 0;

    AM_NODISCARD bool valid() const noexcept {
        return data != nullptr && size != 0;
    }
};

struct RuntimeSequenceState {
    bool prepared = false;
    bool prefill_done = false;
    size_t prompt_len = 0;
    size_t generated_len = 0;
    size_t current_pos = 0;
};

/// Per-step tensor binding passed to the executor before Execute().
///
/// TensorViews borrow data/stride pointers that must remain valid for the
/// entire duration of the Execute() call.
struct StepTensorBinding {
    std::vector<TensorView> inputs;
    std::vector<MutableTensorView> outputs;
};

class RuntimeBindingContext {
public:
    RuntimeBindingContext() = default;
    explicit RuntimeBindingContext(WorkspaceArena* workspace_arena) noexcept
        : workspace_arena_(workspace_arena) {}

    void SetWorkspaceArena(WorkspaceArena* workspace_arena) noexcept;

    AM_NODISCARD WorkspaceArena* GetWorkspaceArena() const noexcept;

    AM_NODISCARD StatusOr<WorkspaceBinding> BindWorkspace(
            const WorkspaceRequirement& requirement) const noexcept;

    void SetKVCacheView(const KVCacheView& kv_cache_view) noexcept;

    AM_NODISCARD const KVCacheView& kv_cache_view() const noexcept;

    AM_NODISCARD bool HasKVCacheView() const noexcept;

    void ClearKVCacheView() noexcept;

    void SetTempBufferBinding(TempBufferKind kind,
                              const TempBufferBinding& binding) noexcept;

    AM_NODISCARD const TempBufferBinding& GetTempBufferBinding(TempBufferKind kind) const noexcept;

    AM_NODISCARD bool HasTempBufferBinding(TempBufferKind kind) const noexcept;

    void ClearTempBufferBinding(TempBufferKind kind) noexcept;

    void ClearAllTempBufferBindings() noexcept;

    AM_NODISCARD RuntimeSequenceState& mutable_sequence_state() noexcept;

    AM_NODISCARD const RuntimeSequenceState& sequence_state() const noexcept;

    void ResetSequenceState() noexcept;

    void SetStepTensorBinding(size_t step_index, StepTensorBinding binding);

    AM_NODISCARD StatusOr<const StepTensorBinding*> GetStepTensorBinding(
            size_t step_index) const noexcept;

    void Reset() noexcept;

private:
    static constexpr size_t ToIndex(TempBufferKind kind) noexcept {
        return static_cast<size_t>(kind);
    }

    WorkspaceArena* workspace_arena_ = nullptr;
    KVCacheView kv_cache_view_{};
    std::array<TempBufferBinding, static_cast<size_t>(TempBufferKind::kCount)> temp_buffers_{};
    RuntimeSequenceState sequence_state_{};
    std::vector<StepTensorBinding> step_tensor_bindings_{};
};

}// namespace aethermind

#endif
