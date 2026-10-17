#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "inference/test_malloc_interposer.h"
#include "test_kv_cache_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

namespace {

using namespace aethermind;

template<typename T>
Status TakeStatus(StatusOr<T> result) noexcept {
    return std::move(result).status();
}

struct DiagnosticCase {
    const char* name;
    StatusCode code;
    Status (*invoke)(const KVCacheSessionView&, const KVCacheLayout&);
};

constexpr DiagnosticCase kDiagnosticCases[]{
        {"WriteEmpty", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateWrite(0, 0, 2, 0); }},
        {"WriteRangeOverflow", StatusCode::kOverflow,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateWrite(0, 0, std::numeric_limits<size_t>::max(), 1); }},
        {"WriteBeyondReservation", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateWrite(0, 0, view.token_capacity(), 1); }},
        {"WriteLayer", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateWrite(view.num_layers(), 0, 2, 1); }},
        {"WriteHead", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateWrite(0, view.num_kv_heads(), 2, 1); }},
        {"ReadReversed", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateRead(0, 0, 2, 1); }},
        {"ReadBeyondCommitted", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateRead(0, 0, 0, view.committed_tokens() + 1); }},
        {"ReadLayer", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateRead(view.num_layers(), 0, 0, 2); }},
        {"ReadHead", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return view.ValidateRead(0, view.num_kv_heads(), 0, 2); }},
        {"AppendWrongBegin", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForAppend(0, 1, 3)); }},
        {"AppendEmpty", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForAppend(0, 2, 2)); }},
        {"AppendBeyondReservation", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForAppend(0, 2, view.token_capacity() + 1)); }},
        {"AppendStorageLayer", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForAppend(view.num_layers(), 2, 3)); }},
        {"ReadBindingWrongCommitted", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForRead(0, 1, 2)); }},
        {"ReadBindingReversedVisible", StatusCode::kInvalidArgument,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForRead(0, 2, 1)); }},
        {"ReadBindingBeyondReservation", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForRead(0, 2, view.token_capacity() + 1)); }},
        {"ReadStorageLayer", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.BindLayerForRead(view.num_layers(), 2, 2)); }},
        {"UnboundWrite", StatusCode::kFailedPrecondition,
         [](const KVCacheSessionView&, const KVCacheLayout&) { return KVCacheSessionView{}.ValidateWrite(0, 0, 0, 1); }},
        {"UnboundReadBinding", StatusCode::kFailedPrecondition,
         [](const KVCacheSessionView&, const KVCacheLayout&) { return TakeStatus(KVCacheSessionView{}.BindLayerForRead(0, 0, 0)); }},
        {"KeyDimension", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.KeyData(0, 0, 0, view.head_dim())); }},
        {"ValueDimension", StatusCode::kOutOfRange,
         [](const KVCacheSessionView& view, const KVCacheLayout&) { return TakeStatus(view.ValueData(0, 0, 0, view.head_dim())); }},
        {"LayoutLayer", StatusCode::kOutOfRange,
         [](const KVCacheSessionView&, const KVCacheLayout& layout) { return TakeStatus(layout.Offset(layout.num_layers, 0, 0, 0)); }},
        {"LayoutHead", StatusCode::kOutOfRange,
         [](const KVCacheSessionView&, const KVCacheLayout& layout) { return TakeStatus(layout.Offset(0, layout.num_kv_heads, 0, 0)); }},
        {"LayoutPosition", StatusCode::kOutOfRange,
         [](const KVCacheSessionView&, const KVCacheLayout& layout) { return TakeStatus(layout.Offset(0, 0, layout.max_tokens, 0)); }},
        {"LayoutDimension", StatusCode::kOutOfRange,
         [](const KVCacheSessionView&, const KVCacheLayout& layout) { return TakeStatus(layout.Offset(0, 0, 0, layout.head_dim)); }},
};

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

class KVCacheSessionViewOomTest : public ::testing::TestWithParam<DiagnosticCase> {
protected:
    void SetUp() override {
        if (!test::MallocInterposerAvailable()) {
            GTEST_SKIP() << "Requires the glibc malloc interposer";
        }
        ASSERT_TRUE(storage_.layout.Validate().ok());
        storage_.key_buffer = MakeTestBuffer(kPlaneBytes);
        storage_.value_buffer = MakeTestBuffer(kPlaneBytes);
        ASSERT_TRUE(storage_.is_initialized());
        std::memset(storage_.key_buffer.mutable_data(), 0x35, kPlaneBytes);
        std::memset(storage_.value_buffer.mutable_data(), 0x7a, kPlaneBytes);
        ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());

        state_before_ = state_;
        key_address_ = storage_.key_buffer.data();
        value_address_ = storage_.value_buffer.data();
        std::memcpy(key_before_.data(), key_address_, kPlaneBytes);
        std::memcpy(value_before_.data(), value_address_, kPlaneBytes);
    }

    void ExpectUnchanged() const {
        EXPECT_EQ(state_.generation, state_before_.generation);
        EXPECT_EQ(state_.capacity_tokens, state_before_.capacity_tokens);
        EXPECT_EQ(state_.prompt_len, state_before_.prompt_len);
        EXPECT_EQ(state_.committed_tokens, state_before_.committed_tokens);
        EXPECT_EQ(state_.lifecycle, state_before_.lifecycle);
        EXPECT_EQ(storage_.key_buffer.data(), key_address_);
        EXPECT_EQ(storage_.value_buffer.data(), value_address_);
        EXPECT_EQ(storage_.key_buffer.nbytes(), kPlaneBytes);
        EXPECT_EQ(storage_.value_buffer.nbytes(), kPlaneBytes);
        EXPECT_EQ(std::memcmp(storage_.key_buffer.data(), key_before_.data(), kPlaneBytes), 0);
        EXPECT_EQ(std::memcmp(storage_.value_buffer.data(), value_before_.data(), kPlaneBytes), 0);
    }

    void ExpectUsable() const {
        EXPECT_TRUE(view_.valid());
        EXPECT_TRUE(view_.ValidateWrite(0, 0, 2, 1).ok());
        EXPECT_TRUE(view_.ValidateRead(0, 0, 0, 2).ok());
        const auto append = view_.BindLayerForAppend(0, 2, 3);
        ASSERT_TRUE(append.ok()) << append.status().ToString();
        EXPECT_EQ(append->storage.key_data, key_address_);
        EXPECT_EQ(append->storage.value_data, value_address_);
        const auto read = view_.BindLayerForRead(0, 2, 2);
        ASSERT_TRUE(read.ok()) << read.status().ToString();
        EXPECT_EQ(read->storage.key_data, key_address_);
        EXPECT_EQ(read->storage.value_data, value_address_);
        const auto key = view_.KeyData(0, 0, 0);
        ASSERT_TRUE(key.ok()) << key.status().ToString();
        EXPECT_EQ(*key, key_address_);
        const auto value = view_.ValueData(0, 0, 0);
        ASSERT_TRUE(value.ok()) << value.status().ToString();
        EXPECT_EQ(*value, value_address_);
    }

    static constexpr size_t kPlaneBytes = 768;
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
            .lifecycle = KVCacheSessionState::Lifecycle::kAwaitingPrefill,
    };
    KVCacheSessionView view_{&storage_, &state_};
    KVCacheSessionState state_before_{};
    const void* key_address_ = nullptr;
    const void* value_address_ = nullptr;
    std::array<std::byte, kPlaneBytes> key_before_{};
    std::array<std::byte, kPlaneBytes> value_before_{};
};

TEST_P(KVCacheSessionViewOomTest, DiagnosticAllocationFailureReturnsResourceExhausted) {
    const auto& diagnostic = GetParam();
    test::BeginMallocCallCounting();
    const Status original = diagnostic.invoke(view_, storage_.layout);
    const auto counts = test::EndMallocCallCounting();
    EXPECT_EQ(original.code(), diagnostic.code);
    EXPECT_FALSE(original.message().empty());
    EXPECT_GT(counts.malloc_calls, 0U);

    // Replay every allocation observed on the normal error path, including
    // propagation copies, without assuming a standard library's string layout.
    for (size_t skip = 0; skip < counts.malloc_calls; ++skip) {
        SCOPED_TRACE(skip);
        const Status before_failure = diagnostic.invoke(view_, storage_.layout);
        EXPECT_EQ(before_failure.code(), diagnostic.code);
        EXPECT_FALSE(before_failure.message().empty());

        test::ScopedMallocFailure failure(0, skip);
        const Status status = diagnostic.invoke(view_, storage_.layout);
        const auto result = failure.Stop();

        EXPECT_TRUE(result.allocation_failed);
        EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
        EXPECT_TRUE(status.message().empty());
        ExpectUnchanged();
        ExpectUsable();
    }

    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 3).ok());
    EXPECT_EQ(view_.current_pos(), 3U);
    EXPECT_EQ(state_.generation, state_before_.generation);
    EXPECT_EQ(state_.capacity_tokens, state_before_.capacity_tokens);
    EXPECT_EQ(state_.prompt_len, state_before_.prompt_len);
    EXPECT_EQ(state_.lifecycle, KVCacheSessionState::Lifecycle::kPrefillCommitted);
    EXPECT_TRUE(view_.KeyData(0, 0, 2).ok());
    EXPECT_TRUE(view_.ValueData(0, 0, 2).ok());
}

INSTANTIATE_TEST_SUITE_P(
        DiagnosticPaths,
        KVCacheSessionViewOomTest,
        ::testing::ValuesIn(kDiagnosticCases),
        [](const ::testing::TestParamInfo<DiagnosticCase>& info) { return std::string(info.param.name); });

} // namespace
