#include "aethermind/memory/memory_handle.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

void ExpectEmptyHandle(const MemoryHandle& handle) {
    EXPECT_FALSE(static_cast<bool>(handle));
    EXPECT_EQ(handle.get(), nullptr);
    EXPECT_EQ(handle.context(), nullptr);
    EXPECT_EQ(handle.deleter(), nullptr);
    EXPECT_EQ(static_cast<int>(handle.device().type()), static_cast<int>(kUndefined));
    EXPECT_EQ(static_cast<int>(handle.device().index()), -1);
    EXPECT_EQ(handle.alignment(), 0U);
}

struct ResetObservation {
    MemoryHandle* owner = nullptr;
    size_t calls = 0;
    void* received_context = nullptr;
    void* received_data = nullptr;
};

void ObserveEmptyOwner(void* context, void* data) noexcept {
    auto& state = *static_cast<ResetObservation*>(context);
    ++state.calls;
    state.received_context = context;
    state.received_data = data;
    ExpectEmptyHandle(*state.owner);
}

void ResetOwnerOnDeletion(void* context, void* data) noexcept {
    auto& state = *static_cast<ResetObservation*>(context);
    state.received_data = data;
    // Limit reentry so a regression reports duplicate deletion without recursion.
    if (++state.calls == 1) {
        state.owner->reset();
    }
}

struct ReplacementContext {
    MemoryHandle* owner = nullptr;
    void* replacement_data = nullptr;
    size_t original_calls = 0;
    size_t replacement_calls = 0;
    void* released_replacement = nullptr;
};

void CountReplacementDeletion(void* context, void* data) noexcept {
    auto& state = *static_cast<ReplacementContext*>(context);
    ++state.replacement_calls;
    state.released_replacement = data;
}

void ReplaceOwnerOnDeletion(void* context, void*) noexcept {
    auto& state = *static_cast<ReplacementContext*>(context);
    if (++state.original_calls == 1) {
        *state.owner = MemoryHandle(state.replacement_data, context, &CountReplacementDeletion, Device(kCUDA, 2), 32);
    }
}

TEST(MemoryHandle, ResetClearsAllMetadataBeforeDeletion) {
    alignas(64) unsigned char data[64]{};
    ResetObservation state;
    MemoryHandle handle(data, &state, &ObserveEmptyOwner, Device::CPU(), 64);
    state.owner = &handle;

    handle.reset();

    EXPECT_EQ(state.calls, 1U);
    EXPECT_EQ(state.received_context, &state);
    EXPECT_EQ(state.received_data, data);
    ExpectEmptyHandle(handle);
}

TEST(MemoryHandle, ReentrantResetDeletesOriginalResourceOnce) {
    unsigned char data = 0;
    ResetObservation state;
    {
        MemoryHandle handle(&data, &state, &ResetOwnerOnDeletion, Device::CPU());
        state.owner = &handle;

        handle.reset();

        EXPECT_EQ(state.calls, 1U);
        EXPECT_EQ(state.received_data, &data);
        ExpectEmptyHandle(handle);
        handle.reset();
        EXPECT_EQ(state.calls, 1U);
    }
    EXPECT_EQ(state.calls, 1U);
}

TEST(MemoryHandle, ExplicitResetPreservesResourceBoundDuringDeletion) {
    alignas(64) unsigned char original_data[64]{};
    alignas(64) unsigned char replacement_data[64]{};
    ReplacementContext state;
    state.replacement_data = replacement_data;
    MemoryHandle handle(original_data, &state, &ReplaceOwnerOnDeletion, Device::CPU(), 64);
    state.owner = &handle;

    handle.reset();

    EXPECT_EQ(state.original_calls, 1U);
    EXPECT_EQ(state.replacement_calls, 0U);
    EXPECT_EQ(handle.get(), replacement_data);
    EXPECT_EQ(handle.context(), &state);
    EXPECT_EQ(handle.deleter(), &CountReplacementDeletion);
    EXPECT_EQ(static_cast<int>(handle.device().type()), static_cast<int>(kCUDA));
    EXPECT_EQ(static_cast<int>(handle.device().index()), 2);
    EXPECT_EQ(handle.alignment(), 32U);

    handle.reset();

    EXPECT_EQ(state.original_calls, 1U);
    EXPECT_EQ(state.replacement_calls, 1U);
    EXPECT_EQ(state.released_replacement, replacement_data);
    ExpectEmptyHandle(handle);
    handle.reset();
    EXPECT_EQ(state.replacement_calls, 1U);
}

} // namespace
