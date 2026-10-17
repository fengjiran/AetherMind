#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "inference/test_malloc_interposer.h"
#include "test_kv_cache_helpers.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <limits>
#include <type_traits>

namespace {

using namespace aethermind;

using Lifecycle = KVCacheSessionState::Lifecycle;

template<typename View>
concept HasPublicCommitUntil = requires(View& view) { view.CommitUntil(size_t{1}); };

template<typename View>
concept HasPublicCommitAppend = requires(View& view) { view.CommitAppend(size_t{0}, size_t{1}); };

template<typename View>
concept HasPublicMutableKeyAccess = requires(View& view) { view.MutableKeyData(size_t{0}, size_t{0}, size_t{0}, size_t{0}); };

template<typename View>
concept HasPublicMutableValueAccess = requires(View& view) { view.MutableValueData(size_t{0}, size_t{0}, size_t{0}, size_t{0}); };

TEST(KVCacheSessionView, ExposesNoPublicCommitMutation) {
    static_assert(!HasPublicCommitUntil<KVCacheSessionView>);
    static_assert(!HasPublicCommitAppend<KVCacheSessionView>);
}

TEST(KVCacheSessionView, ExposesNoPublicElementWriteAccess) {
    static_assert(!HasPublicMutableKeyAccess<KVCacheSessionView>);
    static_assert(!HasPublicMutableValueAccess<KVCacheSessionView>);
}

TEST(KVCacheSessionView, BindsStorageAndSessionWithoutSeparateLayout) {
    static_assert(std::is_nothrow_constructible_v<KVCacheSessionView, KVCacheStorage*, KVCacheSessionState*>);
    static_assert(!std::is_constructible_v<KVCacheSessionView, const KVCacheLayout*, KVCacheStorage*, KVCacheSessionState*>);
}

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, nbytes) != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), 64)};
}

void ExpectStateUnchanged(const KVCacheSessionState& state, const KVCacheSessionState& before) {
    EXPECT_EQ(state.generation, before.generation);
    EXPECT_EQ(state.capacity_tokens, before.capacity_tokens);
    EXPECT_EQ(state.prompt_len, before.prompt_len);
    EXPECT_EQ(state.committed_tokens, before.committed_tokens);
    EXPECT_EQ(state.lifecycle, before.lifecycle);
}

class KVCacheCommitTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(storage_.layout.Validate().ok());
        storage_.key_buffer = MakeTestBuffer(768);
        storage_.value_buffer = MakeTestBuffer(768);
        ASSERT_TRUE(storage_.is_initialized());
    }

    KVCacheStorage storage_{
            .layout = {
                    .num_layers = 2,
                    .num_kv_heads = 3,
                    .max_tokens = 8,
                    .head_dim = 5,
                    .padded_head_dim = 8,
                    .token_stride = 16,
                    .head_stride = 128,
                    .layer_stride = 384,
                    .kv_dtype = DataType(DLDataTypeCode::kFloat, 16, 1),
                    .alignment = 16,
            },
    };
    KVCacheSessionState state_{
            .generation = 1,
            .capacity_tokens = 6,
            .prompt_len = 2,
            .lifecycle = Lifecycle::kAwaitingPrefill,
    };
    KVCacheSessionView view_{&storage_, &state_};
};

TEST_F(KVCacheCommitTest, PartialPrefillDoesNotPublishState) {
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 0, 1).code(), StatusCode::kFailedPrecondition);

    ExpectStateUnchanged(state_, before);
    EXPECT_FALSE(view_.KeyData(0, 0, 0).ok());
}

TEST_F(KVCacheCommitTest, PrefillCannotIncludeDecodePositions) {
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 0, 3).code(), StatusCode::kFailedPrecondition);

    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, NoncontiguousBeginDoesNotPublishState) {
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 1, 3).code(), StatusCode::kInvalidArgument);

    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, DecodeCannotSkipPositions) {
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 2, 4).code(), StatusCode::kInvalidArgument);

    ExpectStateUnchanged(state_, before);
    EXPECT_FALSE(view_.KeyData(0, 0, 2).ok());
}

TEST_F(KVCacheCommitTest, DuplicateCommitDoesNotPublishState) {
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 3).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 2, 3).code(), StatusCode::kInvalidArgument);

    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, EmptyAppendDoesNotPublishState) {
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 2, 2).code(), StatusCode::kInvalidArgument);

    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, AppendBeyondReservationDoesNotPublishState) {
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 6).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 6, 7).code(), StatusCode::kOutOfRange);

    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, WrongEpochCannotChangeTheFreshReservation) {
    ++state_.generation;
    KVCacheSessionView fresh_view(&storage_, &state_);
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(fresh_view, 2).ok());
    const auto before = state_;

    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(view_, 2, 3).code(), StatusCode::kFailedPrecondition);

    ExpectStateUnchanged(state_, before);
    EXPECT_FALSE(view_.valid());
    EXPECT_TRUE(fresh_view.valid());
    EXPECT_EQ(fresh_view.committed_tokens(), 2U);
}

TEST_F(KVCacheCommitTest, SaturatedGenerationDoesNotReviveAnIdleView) {
    state_.generation = std::numeric_limits<uint64_t>::max();
    KVCacheSessionView last_view(&storage_, &state_);
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(last_view, 2).ok());
    EXPECT_TRUE(last_view.valid());

    // Model the terminal idle representation to exercise view liveness at the
    // saturated epoch; this does not stand in for Manager Release/Reserve tests.
    state_.capacity_tokens = 0;
    state_.prompt_len = 0;
    state_.committed_tokens = 0;
    state_.lifecycle = Lifecycle::kIdle;
    const auto before = state_;

    EXPECT_FALSE(last_view.valid());
    EXPECT_EQ(detail::KVCacheCommitAccess::CommitAppend(last_view, 0, 2).code(), StatusCode::kFailedPrecondition);
    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, CommitDiagnosticFailurePreservesTheEntireState) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());
    const auto before = state_;

    test::ScopedMallocFailure failure(0);
    const Status status = detail::KVCacheCommitAccess::CommitAppend(view_, 2, 4);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, StaleCommitDiagnosticFailurePreservesTheEntireState) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    ++state_.generation;
    const auto before = state_;

    test::ScopedMallocFailure failure(0);
    const Status status = detail::KVCacheCommitAccess::CommitAppend(view_, 0, 2);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, MalformedReservationDiagnosticFailurePreservesTheEntireState) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    state_.capacity_tokens = 1;
    const auto before = state_;

    test::ScopedMallocFailure failure(0);
    const Status status = detail::KVCacheCommitAccess::CommitAppend(view_, 0, 2);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    ExpectStateUnchanged(state_, before);
}

TEST_F(KVCacheCommitTest, InvalidLayoutDiagnosticFailurePreservesTheEntireState) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    KVCacheStorage malformed_storage = storage_;
    malformed_storage.layout.token_stride = 0;
    KVCacheSessionView malformed_view(&malformed_storage, &state_);
    const auto before = state_;

    test::ScopedMallocFailure failure(0);
    const Status status = detail::KVCacheCommitAccess::CommitAppend(malformed_view, 0, 2);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    ExpectStateUnchanged(state_, before);
}

struct InvalidSessionState {
    const char* name;
    KVCacheSessionState state;
};

class KVCacheInvalidStateTest : public KVCacheCommitTest,
                                public ::testing::WithParamInterface<InvalidSessionState> {};

TEST_P(KVCacheInvalidStateTest, ReadWriteAndCommitFailClosed) {
    state_ = GetParam().state;
    const auto before = state_;

    EXPECT_FALSE(view_.ValidateWrite(0, 0, 0, 1).ok());
    EXPECT_FALSE(view_.ValidateRead(0, 0, 0, 1).ok());
    EXPECT_FALSE(view_.BindLayerForAppend(0, state_.committed_tokens, state_.committed_tokens + 1).ok());
    EXPECT_FALSE(view_.BindLayerForRead(0, state_.committed_tokens, state_.committed_tokens).ok());
    EXPECT_FALSE(view_.KeyData(0, 0, 0).ok());
    EXPECT_FALSE(view_.ValueData(0, 0, 0).ok());
    EXPECT_FALSE(detail::KVCacheCommitAccess::CommitAppend(view_, state_.committed_tokens, state_.committed_tokens + 1).ok());

    ExpectStateUnchanged(state_, before);
}

INSTANTIATE_TEST_SUITE_P(
        InvalidReservations,
        KVCacheInvalidStateTest,
        ::testing::Values(
                InvalidSessionState{"Idle", {.generation = 1, .lifecycle = Lifecycle::kIdle}},
                InvalidSessionState{"UnknownLifecycle", {.generation = 1, .capacity_tokens = 6, .prompt_len = 2, .lifecycle = static_cast<Lifecycle>(255)}},
                InvalidSessionState{"EmptyPrompt", {.generation = 1, .capacity_tokens = 6, .lifecycle = Lifecycle::kAwaitingPrefill}},
                InvalidSessionState{"PromptPastCapacity", {.generation = 1, .capacity_tokens = 1, .prompt_len = 2, .lifecycle = Lifecycle::kAwaitingPrefill}},
                InvalidSessionState{"CapacityPastStorage", {.generation = 1, .capacity_tokens = 9, .prompt_len = 2, .lifecycle = Lifecycle::kAwaitingPrefill}},
                InvalidSessionState{"AwaitingPrefillWithCommittedTokens", {.generation = 1, .capacity_tokens = 6, .prompt_len = 2, .committed_tokens = 1, .lifecycle = Lifecycle::kAwaitingPrefill}},
                InvalidSessionState{"IncompleteCommittedPrefill", {.generation = 1, .capacity_tokens = 6, .prompt_len = 2, .committed_tokens = 1, .lifecycle = Lifecycle::kPrefillCommitted}},
                InvalidSessionState{"CommittedTokensPastCapacity", {.generation = 1, .capacity_tokens = 6, .prompt_len = 2, .committed_tokens = 7, .lifecycle = Lifecycle::kPrefillCommitted}}),
        [](const ::testing::TestParamInfo<InvalidSessionState>& info) { return info.param.name; });

} // namespace
