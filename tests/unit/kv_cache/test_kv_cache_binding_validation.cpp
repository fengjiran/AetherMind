#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "test_kv_cache_helpers.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

namespace {

using namespace aethermind;

void FreeBindingTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeBindingTestBuffer(size_t nbytes) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, nbytes) != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeBindingTestBuffer, Device::CPU(), 64)};
}

class KVCacheBindingValidationTest : public ::testing::Test {
protected:
    void SetUp() override {
        ASSERT_TRUE(storage_.layout.Validate().ok());
        storage_.key_buffer = MakeBindingTestBuffer(kBufferBytes);
        storage_.value_buffer = MakeBindingTestBuffer(kBufferBytes);
        ASSERT_TRUE(storage_.is_initialized());
        std::memset(storage_.key_buffer.mutable_data(), 0, kBufferBytes);
        std::memset(storage_.value_buffer.mutable_data(), 0, kBufferBytes);
        ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 2).ok());
    }

    void ExpectStorage(const KVCacheLayerStorageBinding& binding, size_t layer) const {
        EXPECT_EQ(binding.key_data, static_cast<const std::byte*>(storage_.key_buffer.data()) + layer * storage_.layout.layer_stride);
        EXPECT_EQ(binding.value_data, static_cast<const std::byte*>(storage_.value_buffer.data()) + layer * storage_.layout.layer_stride);
        EXPECT_EQ(binding.dtype, storage_.layout.kv_dtype);
        EXPECT_EQ(binding.layer_index, layer);
        EXPECT_EQ(binding.num_kv_heads, storage_.layout.num_kv_heads);
        EXPECT_EQ(binding.head_dim, storage_.layout.head_dim);
        EXPECT_EQ(binding.token_capacity, state_.capacity_tokens);
        EXPECT_EQ(binding.token_stride_bytes, storage_.layout.token_stride);
        EXPECT_EQ(binding.head_stride_bytes, storage_.layout.head_stride);
    }

    void ExpectAccessRejected(StatusCode code) const {
        EXPECT_EQ(view_.BindLayerForAppend(2, 2, 3).status().code(), code);
        EXPECT_EQ(view_.BindLayerForRead(2, 2, 2).status().code(), code);
        EXPECT_EQ(view_.ValidateWrite(2, 1, 2, 1).code(), code);
        EXPECT_EQ(view_.ValidateRead(2, 1, 1, 2).code(), code);
        EXPECT_EQ(view_.KeyData(2, 1, 1, 4).status().code(), code);
        EXPECT_EQ(view_.ValueData(2, 1, 1, 4).status().code(), code);
    }

    static constexpr size_t kBufferBytes = 3072;
    KVCacheStorage storage_{
            .layout = {
                    .num_layers = 3,
                    .num_kv_heads = 2,
                    .max_tokens = 8,
                    .head_dim = 5,
                    .padded_head_dim = 8,
                    .token_stride = 32,
                    .head_stride = 288,
                    .layer_stride = 640,
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
};

TEST_F(KVCacheBindingValidationTest, BindingsPreservePaddedLayerGeometry) {
    for (size_t layer = 0; layer < storage_.layout.num_layers; ++layer) {
        SCOPED_TRACE(layer);
        const auto append = view_.BindLayerForAppend(layer, 2, 3);
        ASSERT_TRUE(append.ok()) << append.status().ToString();
        ExpectStorage(append->storage, layer);
        EXPECT_EQ(append->begin, 2U);
        EXPECT_EQ(append->end, 3U);

        const auto read = view_.BindLayerForRead(layer, 2, 3);
        ASSERT_TRUE(read.ok()) << read.status().ToString();
        ExpectStorage(read->storage, layer);
        EXPECT_EQ(read->committed_end, 2U);
        EXPECT_EQ(read->visible_end, 3U);
    }
}

TEST_F(KVCacheBindingValidationTest, ElementReadsPreservePaddedOffsetsAndIndexChecks) {
    const auto key = view_.KeyData(2, 1, 1, 4);
    const auto value = view_.ValueData(2, 1, 1, 4);
    ASSERT_TRUE(key.ok()) << key.status().ToString();
    ASSERT_TRUE(value.ok()) << value.status().ToString();

    const auto* key_base = static_cast<const std::byte*>(storage_.key_buffer.data());
    const auto* value_base = static_cast<const std::byte*>(storage_.value_buffer.data());
    EXPECT_EQ(*key, key_base + 1608);
    EXPECT_EQ(*value, value_base + 1608);
    EXPECT_EQ(view_.BindLayerForAppend(3, 2, 3).status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.ValidateWrite(2, 2, 2, 1).code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.KeyData(3, 1, 1, 4).status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.ValueData(2, 2, 1, 4).status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.KeyData(2, 1, 1, 5).status().code(), StatusCode::kOutOfRange);
    EXPECT_EQ(view_.ValueData(2, 1, 2, 4).status().code(), StatusCode::kOutOfRange);
}

TEST_F(KVCacheBindingValidationTest, RevalidatesLayoutOnEveryCall) {
    const auto original = storage_.layout;
    for (bool invalid_alignment: {false, true}) {
        SCOPED_TRACE(invalid_alignment);
        storage_.layout = original;
        EXPECT_TRUE(view_.BindLayerForAppend(2, 2, 3).ok());
        EXPECT_TRUE(view_.BindLayerForRead(2, 2, 2).ok());
        if (invalid_alignment) {
            storage_.layout.alignment = 3;
        } else {
            storage_.layout.head_stride = storage_.layout.token_stride * storage_.layout.max_tokens - 1;
        }

        ExpectAccessRejected(StatusCode::kInvalidArgument);
        EXPECT_EQ(storage_.layout.Offset(2, 1, 1, 4).status().code(), StatusCode::kInvalidArgument);
        EXPECT_EQ(storage_.layout.BytesPerPlane().status().code(), StatusCode::kInvalidArgument);
    }
}

TEST_F(KVCacheBindingValidationTest, RefreshesBindingsAfterValidLayoutChange) {
    const auto before = view_.BindLayerForAppend(2, 2, 3);
    ASSERT_TRUE(before.ok()) << before.status().ToString();
    storage_.layout.token_stride = 48;
    storage_.layout.head_stride = 416;
    storage_.layout.layer_stride = 896;
    ASSERT_TRUE(storage_.layout.Validate().ok());

    const auto append = view_.BindLayerForAppend(2, 2, 3);
    const auto read = view_.BindLayerForRead(2, 2, 2);
    ASSERT_TRUE(append.ok()) << append.status().ToString();
    ASSERT_TRUE(read.ok()) << read.status().ToString();
    ExpectStorage(append->storage, 2);
    ExpectStorage(read->storage, 2);
    EXPECT_NE(append->storage.key_data, before->storage.key_data);
    EXPECT_NE(append->storage.value_data, before->storage.value_data);
    const auto key = view_.KeyData(2, 1, 1, 4);
    ASSERT_TRUE(key.ok()) << key.status().ToString();
    EXPECT_EQ(*key, static_cast<const std::byte*>(storage_.key_buffer.data()) + 2264);
}

TEST_F(KVCacheBindingValidationTest, RejectsPlaneSizeOverflowAfterLayoutChange) {
    storage_.layout.num_layers = std::numeric_limits<size_t>::max();
    ASSERT_TRUE(storage_.layout.Validate().ok());

    ExpectAccessRejected(StatusCode::kOverflow);
    EXPECT_EQ(storage_.layout.BytesPerPlane().status().code(), StatusCode::kOverflow);
}

TEST_F(KVCacheBindingValidationTest, RevalidatesReplacedBackingBuffers) {
    for (Buffer* buffer: {&storage_.key_buffer, &storage_.value_buffer}) {
        EXPECT_TRUE(view_.BindLayerForAppend(2, 2, 3).ok());
        EXPECT_TRUE(view_.BindLayerForRead(2, 2, 2).ok());
        Buffer original = std::move(*buffer);
        *buffer = MakeBindingTestBuffer(storage_.layout.num_layers * storage_.layout.layer_stride - 1);
        ASSERT_TRUE(buffer->is_initialized());

        ExpectAccessRejected(StatusCode::kFailedPrecondition);

        *buffer = MakeBindingTestBuffer(kBufferBytes);
        ASSERT_TRUE(buffer->is_initialized());
        std::memset(buffer->mutable_data(), 0, kBufferBytes);
        EXPECT_NE(buffer->data(), original.data());
        const auto append = view_.BindLayerForAppend(2, 2, 3);
        const auto read = view_.BindLayerForRead(2, 2, 2);
        ASSERT_TRUE(append.ok()) << append.status().ToString();
        ASSERT_TRUE(read.ok()) << read.status().ToString();
        ExpectStorage(append->storage, 2);
        ExpectStorage(read->storage, 2);
    }
}

TEST_F(KVCacheBindingValidationTest, RechecksCommittedWatermark) {
    EXPECT_TRUE(view_.BindLayerForAppend(2, 2, 3).ok());
    EXPECT_TRUE(view_.BindLayerForRead(2, 2, 2).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(view_, 3).ok());

    EXPECT_EQ(view_.BindLayerForAppend(2, 2, 3).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(view_.BindLayerForRead(2, 2, 3).status().code(), StatusCode::kInvalidArgument);
    EXPECT_TRUE(view_.BindLayerForAppend(2, 3, 4).ok());
    EXPECT_TRUE(view_.BindLayerForRead(2, 3, 3).ok());
}

TEST_F(KVCacheBindingValidationTest, RejectsStaleGenerationAfterSuccessfulBinding) {
    EXPECT_TRUE(view_.BindLayerForAppend(2, 2, 3).ok());
    EXPECT_TRUE(view_.BindLayerForRead(2, 2, 2).ok());
    ++state_.generation;

    ExpectAccessRejected(StatusCode::kFailedPrecondition);
}

TEST_F(KVCacheBindingValidationTest, RejectsReleasedSessionAfterSuccessfulBinding) {
    EXPECT_TRUE(view_.BindLayerForAppend(2, 2, 3).ok());
    EXPECT_TRUE(view_.BindLayerForRead(2, 2, 2).ok());
    state_.lifecycle = KVCacheSessionState::Lifecycle::kIdle;

    ExpectAccessRejected(StatusCode::kFailedPrecondition);
}

} // namespace
