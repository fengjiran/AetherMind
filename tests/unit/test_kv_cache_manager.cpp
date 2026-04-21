#include "aethermind/execution/kv_cache_manager.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

DataType MakeKVType() {
    return DataType(DLDataTypeCode::kFloat, 16, 1);
}

TEST(KVLayoutContract, ComputesOffsetsAndBytesPerPlane) {
    KVCacheLayout layout{
            .num_layers = 2,
            .num_kv_heads = 3,
            .max_tokens = 4,
            .head_dim = 5,
            .head_dim_stride = 8,
            .token_stride = 16,
            .head_stride = 64,
            .layer_stride = 192,
            .kv_dtype = MakeKVType(),
            .alignment = 64,
    };

    const StatusOr<size_t> offset = layout.Offset(1, 2, 3, 4);
    ASSERT_TRUE(offset.ok());
    EXPECT_EQ(offset.value(), 192U + 128U + 48U + 8U);

    const StatusOr<size_t> bytes = layout.BytesPerPlane();
    ASSERT_TRUE(bytes.ok());
    EXPECT_EQ(bytes.value(), 384U);
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
}// namespace aethermind
