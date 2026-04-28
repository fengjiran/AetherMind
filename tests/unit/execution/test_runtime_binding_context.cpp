#include "aethermind/execution/kv_cache_manager.h"
#include "aethermind/execution/runtime_binding_context.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

class RecordingWorkspaceArena final : public WorkspaceArena {
public:
    explicit RecordingWorkspaceArena(WorkspaceBinding binding) noexcept
        : binding_(binding) {}

    WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept override {
        ++bind_calls_;
        last_requirement_ = requirement;
        return binding_;
    }

    void Reset() noexcept override {
        ++reset_calls_;
    }

    int bind_calls() const noexcept {
        return bind_calls_;
    }

    int reset_calls() const noexcept {
        return reset_calls_;
    }

    const WorkspaceRequirement& last_requirement() const noexcept {
        return last_requirement_;
    }

private:
    WorkspaceBinding binding_{};
    WorkspaceRequirement last_requirement_{};
    int bind_calls_ = 0;
    int reset_calls_ = 0;
};

TEST(RuntimeBindingContext, ZeroByteWorkspaceDoesNotRequireArena) {
    RuntimeBindingContext bindings;

    const StatusOr<WorkspaceBinding> binding = bindings.BindWorkspace(
            WorkspaceRequirement{.bytes = 0, .alignment = 64, .offset = 0});

    ASSERT_TRUE(binding.ok());
    EXPECT_EQ(binding->data, nullptr);
    EXPECT_EQ(binding->size, 0U);
}

TEST(RuntimeBindingContext, NonZeroWorkspaceWithoutArenaFails) {
    RuntimeBindingContext bindings;

    const StatusOr<WorkspaceBinding> binding = bindings.BindWorkspace(
            WorkspaceRequirement{.bytes = 32, .alignment = 32, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kFailedPrecondition);
}

TEST(RuntimeBindingContext, BindWorkspaceDelegatesToArena) {
    alignas(64) std::byte storage[128]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    RuntimeBindingContext bindings(&arena);
    const WorkspaceRequirement requirement{.bytes = 64, .alignment = 64, .offset = 32};

    const StatusOr<WorkspaceBinding> binding = bindings.BindWorkspace(requirement);

    ASSERT_TRUE(binding.ok());
    EXPECT_EQ(binding->data, storage);
    EXPECT_EQ(binding->size, sizeof(storage));
    EXPECT_EQ(arena.bind_calls(), 1);
    EXPECT_EQ(arena.last_requirement().bytes, requirement.bytes);
    EXPECT_EQ(arena.last_requirement().alignment, requirement.alignment);
    EXPECT_EQ(arena.last_requirement().offset, requirement.offset);
}

TEST(RuntimeBindingContext, BindWorkspaceRejectsInsufficientBinding) {
    alignas(64) std::byte storage[32]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    RuntimeBindingContext bindings(&arena);

    const StatusOr<WorkspaceBinding> binding = bindings.BindWorkspace(
            WorkspaceRequirement{.bytes = 64, .alignment = 32, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(arena.bind_calls(), 1);
}

TEST(RuntimeBindingContext, ResetDelegatesToArenaAndClearsTransientBindings) {
    RecordingWorkspaceArena arena({});
    RuntimeBindingContext bindings(&arena);
    alignas(32) std::byte scratch[32]{};

    bindings.SetTempBufferBinding(TempBufferKind::kScratch,
                                  TempBufferBinding{.data = scratch, .size = sizeof(scratch)});
    bindings.mutable_sequence_state().prepared = true;
    bindings.mutable_sequence_state().current_pos = 7;

    bindings.Reset();

    EXPECT_EQ(arena.reset_calls(), 1);
    EXPECT_FALSE(bindings.HasTempBufferBinding(TempBufferKind::kScratch));
    EXPECT_FALSE(bindings.sequence_state().prepared);
    EXPECT_EQ(bindings.sequence_state().current_pos, 0U);
}

TEST(RuntimeBindingContext, InvalidAlignmentIsRejectedBeforeArenaBinding) {
    alignas(64) std::byte storage[64]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    RuntimeBindingContext bindings(&arena);

    const StatusOr<WorkspaceBinding> binding = bindings.BindWorkspace(
            WorkspaceRequirement{.bytes = 16, .alignment = 3, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(arena.bind_calls(), 0);
}

TEST(RuntimeBindingContext, KVCacheViewCanBeStoredAndCleared) {
    KVCacheManager manager;
    RuntimeBindingContext bindings;

    ASSERT_TRUE(manager.Init(2,
                             4,
                             128,
                             16,
                             DataType(DLDataTypeCode::kFloat, 16, 1),
                             64)
                        .ok());
    StatusOr<KVCacheView> view = manager.ReserveForSession(12, 8);
    ASSERT_TRUE(view.ok());

    bindings.SetKVCacheView(view.value());

    ASSERT_TRUE(bindings.HasKVCacheView());
    EXPECT_EQ(bindings.kv_cache_view().num_layers(), 2U);
    EXPECT_EQ(bindings.kv_cache_view().num_kv_heads(), 4U);
    EXPECT_EQ(bindings.kv_cache_view().token_capacity(), 20U);
    EXPECT_EQ(bindings.kv_cache_view().committed_tokens(), 12U);

    bindings.ClearKVCacheView();

    EXPECT_FALSE(bindings.HasKVCacheView());
}

TEST(RuntimeBindingContext, TempBufferBindingsCanBeManagedIndependently) {
    alignas(64) std::byte hidden[64]{};
    alignas(64) std::byte logits[128]{};
    RuntimeBindingContext bindings;

    bindings.SetTempBufferBinding(TempBufferKind::kHiddenState,
                                  TempBufferBinding{.data = hidden, .size = sizeof(hidden)});
    bindings.SetTempBufferBinding(TempBufferKind::kLogits,
                                  TempBufferBinding{.data = logits, .size = sizeof(logits)});

    EXPECT_TRUE(bindings.HasTempBufferBinding(TempBufferKind::kHiddenState));
    EXPECT_TRUE(bindings.HasTempBufferBinding(TempBufferKind::kLogits));
    EXPECT_FALSE(bindings.HasTempBufferBinding(TempBufferKind::kScratch));
    EXPECT_EQ(bindings.GetTempBufferBinding(TempBufferKind::kHiddenState).data, hidden);
    EXPECT_EQ(bindings.GetTempBufferBinding(TempBufferKind::kLogits).size, sizeof(logits));

    bindings.ClearTempBufferBinding(TempBufferKind::kHiddenState);

    EXPECT_FALSE(bindings.HasTempBufferBinding(TempBufferKind::kHiddenState));
    EXPECT_TRUE(bindings.HasTempBufferBinding(TempBufferKind::kLogits));
}

TEST(RuntimeBindingContext, SequenceStateCanTrackSessionExecutionProgress) {
    RuntimeBindingContext bindings;

    RuntimeSequenceState& state = bindings.mutable_sequence_state();
    state.prepared = true;
    state.prefill_done = true;
    state.prompt_len = 16;
    state.generated_len = 4;
    state.current_pos = 19;

    EXPECT_TRUE(bindings.sequence_state().prepared);
    EXPECT_TRUE(bindings.sequence_state().prefill_done);
    EXPECT_EQ(bindings.sequence_state().prompt_len, 16U);
    EXPECT_EQ(bindings.sequence_state().generated_len, 4U);
    EXPECT_EQ(bindings.sequence_state().current_pos, 19U);

    bindings.ResetSequenceState();

    EXPECT_FALSE(bindings.sequence_state().prepared);
    EXPECT_FALSE(bindings.sequence_state().prefill_done);
    EXPECT_EQ(bindings.sequence_state().prompt_len, 0U);
    EXPECT_EQ(bindings.sequence_state().generated_len, 0U);
    EXPECT_EQ(bindings.sequence_state().current_pos, 0U);
}

}// namespace
}// namespace aethermind
