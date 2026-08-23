#include "aethermind/runtime/kv_cache_manager.h"

#include <gtest/gtest.h>

#include <cstdlib>

namespace {

using namespace aethermind;

DataType MakeKVType() {
    return DataType(DLDataTypeCode::kFloat, 16, 1);
}

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

KVCacheLayout MakeTestLayout(size_t token_stride = 16,
                             size_t head_stride = 64,
                             size_t layer_stride = 192,
                             size_t alignment = 16,
                             size_t head_dim_stride = 8) {
    return KVCacheLayout{
            .num_layers = 2,
            .num_kv_heads = 3,
            .max_tokens = 4,
            .head_dim = 5,
            .head_dim_stride = head_dim_stride,
            .token_stride = token_stride,
            .head_stride = head_stride,
            .layer_stride = layer_stride,
            .kv_dtype = MakeKVType(),
            .alignment = alignment,
    };
}

TEST(KVLayoutContract, ComputesOffsetsAndBytesPerPlane) {
    const KVCacheLayout layout = MakeTestLayout();

    const StatusOr<size_t> offset = layout.Offset(1, 2, 3, 4);
    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(offset.value(), 192U + 128U + 48U + 8U);

    const StatusOr<size_t> bytes = layout.BytesPerPlane();
    ASSERT_TRUE(bytes.ok());
    EXPECT_EQ(bytes.value(), 384U);
}

TEST(KVLayoutContract, RejectsStridesInconsistentWithGeometry) {
    // token_stride must be a multiple of the element size.
    auto bad_element = MakeTestLayout();
    bad_element.token_stride = 15;
    EXPECT_EQ(bad_element.Validate().code(), StatusCode::kInvalidArgument);

    // token_stride must be aligned to the layout alignment.
    auto unaligned = MakeTestLayout(/*token_stride=*/8, /*head_stride=*/32,
                                    /*layer_stride=*/96);
    EXPECT_EQ(unaligned.Validate().code(), StatusCode::kInvalidArgument);

    // token_stride must cover the full head row.
    auto overlapping_row = MakeTestLayout(/*token_stride=*/8, /*head_stride=*/32,
                                          /*layer_stride=*/96,
                                          /*alignment=*/8);
    EXPECT_EQ(overlapping_row.Validate().code(), StatusCode::kInvalidArgument);

    // head_stride must cover all tokens of a head.
    auto short_head = MakeTestLayout();
    short_head.head_stride = 32;
    short_head.layer_stride = 96;
    EXPECT_EQ(short_head.Validate().code(), StatusCode::kInvalidArgument);

    // layer_stride must cover all heads of a layer.
    auto short_layer = MakeTestLayout();
    short_layer.layer_stride = 128;
    EXPECT_EQ(short_layer.Validate().code(), StatusCode::kInvalidArgument);
}

TEST(KVCacheManager, InitAndReserveCreatesValidView) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(2, 4, 32, 16, MakeKVType(), 64).ok());

    const StatusOr<KVCacheView> view = manager.ReserveForSession(8, 8);

    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->num_layers(), 2U);
    EXPECT_EQ(view->num_kv_heads(), 4U);
    EXPECT_EQ(view->head_dim(), 16U);
    EXPECT_EQ(view->max_tokens(), 32U);
    EXPECT_EQ(view->token_capacity(), 16U);
    EXPECT_EQ(view->current_pos(), 8U);
    EXPECT_GT(manager.total_bytes(), 0U);
}

TEST(KVCacheManager, InitAlignsTokenRowsToLayoutAlignment) {
    // head_dim=16 fp16 with 64-byte alignment: head_dim_stride is padded to
    // 32 elements so every token row (64 bytes) starts at an alignment
    // boundary; a non-divisible head_dim is rounded up the same way.
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(2, 4, 32, 16, MakeKVType(), 64).ok());
    EXPECT_EQ(manager.layout().head_dim_stride, 32U);
    EXPECT_EQ(manager.layout().token_stride, 64U);
    EXPECT_EQ(manager.layout().token_stride % manager.layout().alignment, 0U);
    EXPECT_GE(manager.layout().head_dim_stride, manager.layout().head_dim);

    ASSERT_TRUE(manager.Init(1, 1, 16, 20, MakeKVType(), 64).ok());
    EXPECT_EQ(manager.layout().head_dim_stride, 32U);
    EXPECT_EQ(manager.layout().token_stride, 64U);
    EXPECT_EQ(manager.layout().token_stride % manager.layout().alignment, 0U);
}

TEST(KVCacheView, RejectsStorageSmallerThanLayout) {
    // BytesPerPlane == 1024 for this geometry; back it with 512-byte buffers.
    const KVCacheLayout layout{
            .num_layers = 1,
            .num_kv_heads = 1,
            .max_tokens = 16,
            .head_dim = 8,
            .head_dim_stride = 32,
            .token_stride = 64,
            .head_stride = 1024,
            .layer_stride = 1024,
            .kv_dtype = MakeKVType(),
            .alignment = 64,
    };
    KVCacheStorage storage{
            .key_buffer = MakeTestBuffer(512),
            .value_buffer = MakeTestBuffer(512),
            .kv_dtype = MakeKVType(),
            .alignment = 64,
    };
    SessionKVSlot slot{.generation = 1,
                       .in_use = true,
                       .capacity_tokens = 8,
                       .prompt_len = 4,
                       .current_pos = 4};
    KVCacheView view(&layout, &storage, &slot);

    const auto key = view.KeyData(0, 0, 0);

    ASSERT_FALSE(key.ok());
    EXPECT_EQ(key.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KVCacheManager, ReserveRejectsSecondActiveSession) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(manager.ReserveForSession(4, 4).ok());

    const StatusOr<KVCacheView> second = manager.ReserveForSession(2, 2);

    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KVCacheManager, CommitAndReadWritePointersRespectBounds) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(1, 1, 16, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheView> view = manager.ReserveForSession(4, 4);
    ASSERT_TRUE(view.ok());

    EXPECT_FALSE(view->KeyData(0, 0, 4).ok());

    ASSERT_TRUE(view->CommitUntil(5).ok());
    EXPECT_EQ(view->current_pos(), 5U);

    const StatusOr<void*> key_ptr = view->MutableKeyData(0, 0, 4);
    const StatusOr<void*> value_ptr = view->MutableValueData(0, 0, 4);
    ASSERT_TRUE(key_ptr.ok());
    ASSERT_TRUE(value_ptr.ok());
    EXPECT_NE(key_ptr.value(), value_ptr.value());

    const StatusOr<const void*> key_read = view->KeyData(0, 0, 4);
    ASSERT_TRUE(key_read.ok());
    EXPECT_EQ(key_read.value(), key_ptr.value());
}

TEST(KVCacheManager, ResetSessionRewindsToPromptLength) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(1, 1, 32, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheView> view = manager.ReserveForSession(6, 10);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(view->CommitUntil(10).ok());

    ASSERT_TRUE(manager.ResetSession(*view).ok());

    EXPECT_EQ(view->current_pos(), 6U);
}

TEST(KVCacheManager, ReleaseInvalidatesViewAndAllowsNewReservation) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(1, 2, 24, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheView> view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(view.ok());

    ASSERT_TRUE(manager.ReleaseSession(*view).ok());
    EXPECT_FALSE(view->valid());

    StatusOr<KVCacheView> next = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(next.ok());
    EXPECT_TRUE(next->valid());
    EXPECT_EQ(next->current_pos(), 2U);
}

TEST(KVCacheManager, ReserveRejectsRequestsBeyondPhysicalCapacity) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(1, 1, 8, 8, MakeKVType(), 64).ok());

    const StatusOr<KVCacheView> view = manager.ReserveForSession(6, 4);

    ASSERT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kOutOfRange);
}

}// namespace
