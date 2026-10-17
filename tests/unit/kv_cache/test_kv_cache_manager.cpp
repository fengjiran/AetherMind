#include "aethermind/kv_cache/detail/kv_cache_state.h"
#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "inference/test_malloc_interposer.h"
#include "test_kv_cache_helpers.h"
#include "test_utils/test_allocator.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

namespace {

using namespace aethermind;

static_assert(!std::is_copy_constructible_v<KVCacheManager>);
static_assert(!std::is_copy_assignable_v<KVCacheManager>);
static_assert(std::is_nothrow_move_constructible_v<KVCacheManager>);
static_assert(std::is_nothrow_move_assignable_v<KVCacheManager>);

DataType MakeKVType() {
    return DataType(DLDataTypeCode::kFloat, 16, 1);
}

void ExpectEmptyManager(const KVCacheManager& manager) {
    EXPECT_FALSE(manager.is_initialized());
    EXPECT_EQ(manager.capacity_tokens(), 0U);
    EXPECT_EQ(manager.total_bytes(), 0U);
    const auto& layout = manager.layout();
    EXPECT_EQ(layout.num_layers, 0U);
    EXPECT_EQ(layout.num_kv_heads, 0U);
    EXPECT_EQ(layout.max_tokens, 0U);
    EXPECT_EQ(layout.head_dim, 0U);
    EXPECT_EQ(layout.padded_head_dim, 0U);
    EXPECT_EQ(layout.token_stride, 0U);
    EXPECT_EQ(layout.head_stride, 0U);
    EXPECT_EQ(layout.layer_stride, 0U);
    EXPECT_EQ(layout.kv_dtype, DataType{});
    EXPECT_EQ(layout.alignment, KVCacheLayout{}.alignment);
}

void ExpectEmptyManagerCanRetry(KVCacheManager& manager,
                                const KVCacheSessionView& old_view) {
    ExpectEmptyManager(manager);
    EXPECT_EQ(manager.ReserveForSession(1, 1).status().code(),
              StatusCode::kFailedPrecondition);
    EXPECT_FALSE(old_view.valid());

    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 2, 3, 24, 8, MakeKVType(), 16).ok());
    EXPECT_EQ(manager.capacity_tokens(), 24U);
    EXPECT_GT(manager.total_bytes(), 0U);
    auto fresh = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(fresh.ok());
    EXPECT_TRUE(fresh->valid());
    EXPECT_EQ(fresh->num_layers(), 2U);
    EXPECT_EQ(fresh->num_kv_heads(), 3U);
    EXPECT_FALSE(old_view.valid());
    EXPECT_EQ(manager.ResetSession(old_view).code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(fresh->valid());
    EXPECT_TRUE(manager.ReleaseSession(*fresh).ok());
}

struct InitFailureCase {
    const char* name;
    size_t num_layers = 1;
    size_t num_kv_heads = 1;
    size_t max_tokens = 16;
    size_t head_dim = 8;
    DataType dtype = MakeKVType();
    size_t alignment = 64;
    StatusCode expected_code = StatusCode::kInvalidArgument;
};

class KVCacheManagerInitFailure : public testing::TestWithParam<InitFailureCase> {};

TEST_P(KVCacheManagerInitFailure, LeavesEmptyStateAndCanRetry) {
    const auto& param = GetParam();
    for (const bool reinitialize: {false, true}) {
        SCOPED_TRACE(reinitialize);
        KVCacheManager manager;
        KVCacheSessionView old_view;
        if (reinitialize) {
            ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 8, 8, MakeKVType(), 64).ok());
            auto reservation = manager.ReserveForSession(2, 2);
            ASSERT_TRUE(reservation.ok());
            old_view = *reservation;
        }

        const Status status = manager.Init(test::TestCpuAllocator(), param.num_layers,
                                           param.num_kv_heads, param.max_tokens,
                                           param.head_dim, param.dtype, param.alignment);

        EXPECT_EQ(status.code(), param.expected_code);
        ExpectEmptyManagerCanRetry(manager, old_view);
    }
}

constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();

INSTANTIATE_TEST_SUITE_P(
        Validation, KVCacheManagerInitFailure,
        testing::Values(
                InitFailureCase{.name = "UndefinedDtype", .dtype = {}},
                InitFailureCase{.name = "ZeroLayers", .num_layers = 0},
                InitFailureCase{.name = "ZeroHeads", .num_kv_heads = 0},
                InitFailureCase{.name = "ZeroTokens", .max_tokens = 0},
                InitFailureCase{.name = "ZeroHeadDimension", .head_dim = 0},
                InitFailureCase{.name = "ZeroAlignment", .alignment = 0},
                InitFailureCase{.name = "NonPowerOfTwoAlignment", .alignment = 3},
                InitFailureCase{.name = "TokenStrideOverflow", .head_dim = kMaxSize / 2 + 1, .alignment = 2, .expected_code = StatusCode::kOverflow},
                InitFailureCase{.name = "HeadStrideOverflow", .max_tokens = kMaxSize / 64 + 1, .expected_code = StatusCode::kOverflow},
                InitFailureCase{.name = "LayerStrideOverflow", .num_kv_heads = kMaxSize / 64 + 1, .max_tokens = 1, .expected_code = StatusCode::kOverflow},
                InitFailureCase{.name = "PlaneBytesOverflow", .num_layers = kMaxSize / 64 + 1, .max_tokens = 1, .expected_code = StatusCode::kOverflow},
                InitFailureCase{.name = "TotalBytesOverflow", .num_layers = kMaxSize / 128 + 1, .max_tokens = 1, .expected_code = StatusCode::kOverflow}),
        [](const testing::TestParamInfo<InitFailureCase>& info) { return info.param.name; });

struct InitAllocationFailureCase {
    const char* name;
    test::MallocFailureTarget target;
    size_t skip_matching_allocations;
};

class KVCacheManagerInitAllocationFailure : public testing::TestWithParam<InitAllocationFailureCase> {};

TEST_P(KVCacheManagerInitAllocationFailure, LeavesEmptyStateAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const auto& param = GetParam();
    auto& allocator = test::TestCpuAllocator();
    for (const bool reinitialize: {false, true}) {
        SCOPED_TRACE(reinitialize);
        KVCacheManager manager;
        KVCacheSessionView old_view;
        if (reinitialize) {
            ASSERT_TRUE(manager.Init(allocator, 1, 1, 8, 8, MakeKVType(), 64).ok());
            auto reservation = manager.ReserveForSession(2, 2);
            ASSERT_TRUE(reservation.ok());
            old_view = *reservation;
        }

        const bool metadata_failure = param.target == test::MallocFailureTarget::kMalloc;
        test::ScopedMallocFailure failure(metadata_failure ? sizeof(BufferImpl) : 0,
                                          param.skip_matching_allocations, false, param.target);
        const Status status = manager.Init(allocator, 2, 3, 16, 8, MakeKVType(), 64);
        const auto result = failure.Stop();

        EXPECT_TRUE(result.allocation_failed);
        EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
        const size_t allocated_planes = param.skip_matching_allocations + (metadata_failure ? 1 : 0);
        EXPECT_EQ(result.aligned_allocation_calls, allocated_planes);
        if (allocated_planes != 0) {
            EXPECT_TRUE(result.last_aligned_allocation_released);
        }
        ExpectEmptyManagerCanRetry(manager, old_view);
    }
}

INSTANTIATE_TEST_SUITE_P(
        Allocation, KVCacheManagerInitAllocationFailure,
        testing::Values(
                InitAllocationFailureCase{"FirstPlaneStorage", test::MallocFailureTarget::kPosixMemalign, 0},
                InitAllocationFailureCase{"SecondPlaneStorage", test::MallocFailureTarget::kPosixMemalign, 1},
                InitAllocationFailureCase{"FirstPlaneMetadata", test::MallocFailureTarget::kMalloc, 0},
                InitAllocationFailureCase{"SecondPlaneMetadata", test::MallocFailureTarget::kMalloc, 1}),
        [](const testing::TestParamInfo<InitAllocationFailureCase>& info) { return info.param.name; });

TEST(KVCacheManager, InitDiagnosticAllocationFailureLeavesEmptyStateAndCanRetry) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    auto& allocator = test::TestCpuAllocator();
    for (const bool reinitialize: {false, true}) {
        SCOPED_TRACE(reinitialize);
        KVCacheManager manager;
        KVCacheSessionView old_view;
        if (reinitialize) {
            ASSERT_TRUE(manager.Init(allocator, 1, 1, 8, 8, MakeKVType(), 64).ok());
            auto reservation = manager.ReserveForSession(2, 2);
            ASSERT_TRUE(reservation.ok());
            old_view = *reservation;
        }

        test::ScopedMallocFailure failure(0);
        const Status status = manager.Init(allocator, 1, 1, 16, 8, DataType{}, 64);
        const auto result = failure.Stop();

        EXPECT_TRUE(result.allocation_failed);
        EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
        EXPECT_TRUE(status.message().empty());
        ExpectEmptyManagerCanRetry(manager, old_view);
    }
}

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

// Frees the original 64-byte-aligned block. Used with an interior pointer so
// the handle's data pointer differs from the block the deleter releases.
void FreeOwnedBaseBlock(void* block, void*) noexcept {
    std::free(block);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

// Allocates a 64-byte-aligned block and returns an interior pointer exactly 16
// bytes past it: 16-byte aligned by construction, but provably not 64-byte
// aligned. The declared 16-byte alignment passes BufferImpl's self-check while
// breaking a 64-byte layout row alignment.
Buffer MakeMisalignedBaseTestBuffer(size_t nbytes) {
    void* raw = nullptr;
    if (posix_memalign(&raw, 64, nbytes + 64) != 0 || raw == nullptr) {
        return {};
    }
    void* const ptr = static_cast<std::byte*>(raw) + 16;
    return Buffer{nbytes, MemoryHandle(ptr, raw, &FreeOwnedBaseBlock, Device::CPU(), 16)};
}

// Backs the buffer with a 64-byte-aligned block but declares only 16 bytes,
// exercising the fact-based base check against an under-declared handle.
Buffer MakeUnderDeclaredBaseTestBuffer(size_t nbytes) {
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 64, nbytes) != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), 16)};
}

KVCacheLayout MakeTestLayout(size_t token_stride = 16,
                             size_t head_stride = 64,
                             size_t layer_stride = 192,
                             size_t alignment = 16,
                             size_t padded_head_dim = 8) {
    return KVCacheLayout{
            .num_layers = 2,
            .num_kv_heads = 3,
            .max_tokens = 4,
            .head_dim = 5,
            .padded_head_dim = padded_head_dim,
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
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 2, 4, 32, 16, MakeKVType(), 64).ok());

    const StatusOr<KVCacheSessionView> view = manager.ReserveForSession(8, 8);

    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->num_layers(), 2U);
    EXPECT_EQ(view->num_kv_heads(), 4U);
    EXPECT_EQ(view->head_dim(), 16U);
    EXPECT_EQ(view->max_tokens(), 32U);
    EXPECT_EQ(view->token_capacity(), 16U);
    EXPECT_EQ(view->current_pos(), 0U);
    EXPECT_TRUE(view->awaiting_prefill());
    EXPECT_EQ(view->prompt_len(), 8U);
    EXPECT_GT(manager.total_bytes(), 0U);
}

TEST(KVCacheManager, InitPromotesAlignmentBelowPointerSize) {
    KVCacheManager manager;

    // A layout alignment below sizeof(void*) is a floor the allocator promotes
    // rather than rejects, so Init succeeds. The row-layout guarantee is stated
    // against the requested alignment, not the stronger one actually obtained.
    const size_t requested = sizeof(void*) / 2;
    const Status status = manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8,
                                       MakeKVType(), requested);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(manager.is_initialized());
    EXPECT_GT(manager.total_bytes(), 0U);
    EXPECT_EQ(manager.layout().alignment, requested);
    EXPECT_EQ(manager.layout().token_stride % requested, 0U);

    auto view = manager.ReserveForSession(2, 2);
    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
}

TEST(KVCacheManager, MoveBeforeBorrowingPreservesStorageForReservation) {
    KVCacheManager source;
    ASSERT_TRUE(source.Init(test::TestCpuAllocator(), 2, 4, 32, 16, MakeKVType(), 64).ok());
    const size_t total_bytes = source.total_bytes();

    KVCacheManager manager(std::move(source));
    auto view = manager.ReserveForSession(8, 8);

    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->token_capacity(), 16U);
    EXPECT_EQ(manager.total_bytes(), total_bytes);
    EXPECT_TRUE(manager.ReleaseSession(*view).ok());
}

TEST(KVCacheManager, MoveAssignmentBeforeBorrowingPreservesStorageForReservation) {
    KVCacheManager source;
    ASSERT_TRUE(source.Init(test::TestCpuAllocator(), 2, 4, 32, 16, MakeKVType(), 64).ok());
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 8, 8, MakeKVType(), 64).ok());
    const size_t total_bytes = source.total_bytes();

    manager = std::move(source);
    auto view = manager.ReserveForSession(8, 8);

    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->token_capacity(), 16U);
    EXPECT_EQ(manager.total_bytes(), total_bytes);
    EXPECT_TRUE(manager.ReleaseSession(*view).ok());
}

TEST(KVCacheManager, BufferMetadataFailureReturnsResourceExhaustedAndReleasesPayload) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    KVCacheManager manager;

    test::ScopedMallocFailure failure(sizeof(BufferImpl));
    const Status status = manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64);
    const auto result = failure.Stop();

    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_FALSE(manager.is_initialized());
    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(result.aligned_allocation_calls, 1U);
    EXPECT_TRUE(result.last_aligned_allocation_released);
    EXPECT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
}

TEST(KVCacheManager, InitAlignsTokenRowsToLayoutAlignment) {
    // head_dim=16 fp16 with 64-byte alignment: padded_head_dim is 32
    // elements so every token row (64 bytes) starts at an alignment
    // boundary; a non-divisible head_dim is rounded up the same way.
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 2, 4, 32, 16, MakeKVType(), 64).ok());
    EXPECT_EQ(manager.layout().padded_head_dim, 32U);
    EXPECT_EQ(manager.layout().token_stride, 64U);
    EXPECT_EQ(manager.layout().token_stride % manager.layout().alignment, 0U);
    EXPECT_GE(manager.layout().padded_head_dim, manager.layout().head_dim);

    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 20, MakeKVType(), 64).ok());
    EXPECT_EQ(manager.layout().padded_head_dim, 32U);
    EXPECT_EQ(manager.layout().token_stride, 64U);
    EXPECT_EQ(manager.layout().token_stride % manager.layout().alignment, 0U);
}

TEST(KVCacheSessionView, RejectsStorageSmallerThanLayout) {
    // BytesPerPlane == 1024 for this geometry; back it with 512-byte buffers.
    KVCacheStorage storage{
            .layout = {
                    .num_layers = 1,
                    .num_kv_heads = 1,
                    .max_tokens = 16,
                    .head_dim = 8,
                    .padded_head_dim = 32,
                    .token_stride = 64,
                    .head_stride = 1024,
                    .layer_stride = 1024,
                    .kv_dtype = MakeKVType(),
                    .alignment = 64,
            },
            .key_buffer = MakeTestBuffer(512),
            .value_buffer = MakeTestBuffer(512),
    };
    KVCacheSessionState slot{.generation = 1,
                             .capacity_tokens = 8,
                             .prompt_len = 4,
                             .committed_tokens = 4,
                             .lifecycle = KVCacheSessionState::Lifecycle::kPrefillCommitted};
    KVCacheSessionView view(&storage, &slot);

    const auto key = view.KeyData(0, 0, 0);

    ASSERT_FALSE(key.ok());
    EXPECT_EQ(key.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KVCacheSessionView, RejectsCpuPlaneBaseBreakingRowAlignment) {
    // Mirrors RejectsStorageSmallerThanLayout but keeps the buffers large
    // enough; only the plane base alignment violates the layout contract.
    KVCacheStorage storage{
            .layout = {
                    .num_layers = 1,
                    .num_kv_heads = 1,
                    .max_tokens = 16,
                    .head_dim = 8,
                    .padded_head_dim = 32,
                    .token_stride = 64,
                    .head_stride = 1024,
                    .layer_stride = 1024,
                    .kv_dtype = MakeKVType(),
                    .alignment = 64,
            },
            .key_buffer = MakeMisalignedBaseTestBuffer(1024),
            .value_buffer = MakeMisalignedBaseTestBuffer(1024),
    };
    ASSERT_TRUE(storage.layout.Validate().ok());
    ASSERT_TRUE(storage.is_initialized());
    // Over-allocation guarantees every base lands exactly 16 bytes past a
    // 64-byte boundary, so the fixture is deterministic on any malloc.
    ASSERT_EQ(reinterpret_cast<uintptr_t>(storage.key_buffer.data()) % 64U, 16U);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(storage.value_buffer.data()) % 64U, 16U);
    KVCacheSessionState slot{.generation = 1,
                             .capacity_tokens = 8,
                             .prompt_len = 4,
                             .committed_tokens = 4,
                             .lifecycle = KVCacheSessionState::Lifecycle::kPrefillCommitted};
    KVCacheSessionView view(&storage, &slot);

    const auto key = view.KeyData(0, 0, 0);

    ASSERT_FALSE(key.ok());
    EXPECT_EQ(key.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KVCacheSessionView, AcceptsUnderDeclaredBasesWhenActuallyAligned) {
    // The layout contract checks the base addresses, not the declared minimum
    // alignment, so a weakly declared but actually aligned plane is accepted.
    KVCacheStorage storage{
            .layout = {
                    .num_layers = 1,
                    .num_kv_heads = 1,
                    .max_tokens = 16,
                    .head_dim = 8,
                    .padded_head_dim = 32,
                    .token_stride = 64,
                    .head_stride = 1024,
                    .layer_stride = 1024,
                    .kv_dtype = MakeKVType(),
                    .alignment = 64,
            },
            .key_buffer = MakeUnderDeclaredBaseTestBuffer(1024),
            .value_buffer = MakeUnderDeclaredBaseTestBuffer(1024),
    };
    ASSERT_TRUE(storage.layout.Validate().ok());
    ASSERT_TRUE(storage.is_initialized());
    EXPECT_EQ(storage.key_buffer.alignment(), 16U);
    EXPECT_EQ(storage.key_buffer.device().type(), DeviceType::kCPU);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(storage.key_buffer.data()) % 64U, 0U);
    KVCacheSessionState slot{.generation = 1,
                             .capacity_tokens = 8,
                             .prompt_len = 4,
                             .committed_tokens = 4,
                             .lifecycle = KVCacheSessionState::Lifecycle::kPrefillCommitted};
    KVCacheSessionView view(&storage, &slot);

    EXPECT_TRUE(view.ValidateWrite(0, 0, 4, 1).ok());
    EXPECT_TRUE(view.KeyData(0, 0, 0).ok());
    EXPECT_TRUE(view.BindLayerForRead(0, 4, 4).ok());
}

TEST(KVCacheManager, ReserveRejectsSecondActiveSession) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(manager.ReserveForSession(4, 4).ok());

    const StatusOr<KVCacheSessionView> second = manager.ReserveForSession(2, 2);

    ASSERT_FALSE(second.ok());
    EXPECT_EQ(second.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KVCacheManager, OccupiedReserveDiagnosticFailurePreservesReservation) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto view = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 2).ok());

    test::ScopedMallocFailure failure(0);
    const auto second = manager.ReserveForSession(1, 1);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(second.status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(second.status().message().empty());
    EXPECT_TRUE(view->valid());
    EXPECT_EQ(view->current_pos(), 2U);
    EXPECT_EQ(view->token_capacity(), 6U);
    EXPECT_FALSE(view->awaiting_prefill());
    EXPECT_FALSE(manager.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(manager.ReleaseSession(*view).ok());
}

TEST(KVCacheManager, CommitAndReadWritePointersRespectBounds) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheSessionView> view = manager.ReserveForSession(4, 4);
    ASSERT_TRUE(view.ok());

    const auto append = view->BindLayerForAppend(0, 0, 4);
    ASSERT_TRUE(append.ok()) << append.status().ToString();
    EXPECT_NE(append->storage.key_data, append->storage.value_data);

    EXPECT_FALSE(view->KeyData(0, 0, 0).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 4).ok());
    EXPECT_EQ(view->current_pos(), 4U);
    EXPECT_FALSE(view->awaiting_prefill());

    const StatusOr<const void*> key_read = view->KeyData(0, 0, 0);
    const StatusOr<const void*> value_read = view->ValueData(0, 0, 0);
    ASSERT_TRUE(key_read.ok());
    ASSERT_TRUE(value_read.ok());
    EXPECT_EQ(key_read.value(), append->storage.key_data);
    EXPECT_EQ(value_read.value(), append->storage.value_data);
    EXPECT_FALSE(view->KeyData(0, 0, 4).ok());
    EXPECT_FALSE(view->ValueData(0, 0, 4).ok());
    EXPECT_FALSE(view->BindLayerForAppend(0, 0, 4).ok());
}

TEST(KVCacheManager, ResetSessionClearsCommittedPosition) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 32, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheSessionView> view = manager.ReserveForSession(6, 10);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 6).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 10).ok());

    ASSERT_TRUE(manager.ResetSession(*view).ok());

    EXPECT_EQ(view->current_pos(), 0U);
    EXPECT_EQ(view->prompt_len(), 6U);
    EXPECT_EQ(view->token_capacity(), 16U);
    EXPECT_TRUE(view->awaiting_prefill());
}

TEST(KVCacheManager, ResetKeepsEveryViewCopyLiveInTheSameReservation) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 2, 3, 16, 8, MakeKVType(), 64).ok());
    auto view = manager.ReserveForSession(3, 4);
    ASSERT_TRUE(view.ok());
    KVCacheSessionView copy = *view;
    KVCacheSessionView another_copy = copy;
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 5).ok());

    ASSERT_TRUE(manager.ResetSession(copy).ok());

    for (const auto* current: {&*view, &copy, &another_copy}) {
        EXPECT_TRUE(current->valid());
        EXPECT_EQ(current->current_pos(), 0U);
        EXPECT_EQ(current->committed_tokens(), 0U);
        EXPECT_EQ(current->prompt_len(), 3U);
        EXPECT_EQ(current->token_capacity(), 7U);
        EXPECT_TRUE(current->awaiting_prefill());
    }
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(another_copy, 3).ok());
    EXPECT_EQ(view->committed_tokens(), 3U);
    EXPECT_EQ(copy.committed_tokens(), 3U);
    EXPECT_FALSE(view->awaiting_prefill());
    EXPECT_FALSE(copy.awaiting_prefill());
}

TEST(KVCacheManager, ResetRequiresTheEntirePrefillBeforeDecode) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto view = manager.ReserveForSession(3, 4);
    ASSERT_TRUE(view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 5).ok());
    ASSERT_TRUE(manager.ResetSession(*view).ok());

    EXPECT_FALSE(detail::KVCacheCommitAccess::CommitAppend(*view, 0, 1).ok());
    EXPECT_EQ(view->committed_tokens(), 0U);
    EXPECT_TRUE(view->awaiting_prefill());
    EXPECT_FALSE(view->KeyData(0, 0, 0).ok());

    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 3).ok());
    EXPECT_EQ(view->committed_tokens(), 3U);
    EXPECT_FALSE(view->awaiting_prefill());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 4).ok());
    EXPECT_EQ(view->committed_tokens(), 4U);
}

TEST(KVCacheManager, ReleasedCopyCannotCommitIntoANewReservation) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto view = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(view.ok());
    KVCacheSessionView copy = *view;
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*view, 2).ok());
    ASSERT_TRUE(manager.ReleaseSession(*view).ok());

    EXPECT_FALSE(detail::KVCacheCommitAccess::CommitAppend(copy, 2, 3).ok());
    auto fresh = manager.ReserveForSession(3, 2);
    ASSERT_TRUE(fresh.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*fresh, 3).ok());

    EXPECT_FALSE(detail::KVCacheCommitAccess::CommitAppend(copy, 3, 4).ok());
    EXPECT_FALSE(copy.valid());
    EXPECT_TRUE(fresh->valid());
    EXPECT_EQ(fresh->committed_tokens(), 3U);
    EXPECT_EQ(fresh->prompt_len(), 3U);
    EXPECT_EQ(fresh->token_capacity(), 5U);
    EXPECT_FALSE(fresh->awaiting_prefill());
}

TEST(KVCacheManager, ResetRejectsForeignViewWithoutChangingEitherReservation) {
    KVCacheManager owner;
    KVCacheManager other;
    ASSERT_TRUE(owner.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(other.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto owner_view = owner.ReserveForSession(2, 4);
    auto other_view = other.ReserveForSession(3, 5);
    ASSERT_TRUE(owner_view.ok());
    ASSERT_TRUE(other_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*owner_view, 2).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*other_view, 3).ok());

    const Status status = other.ResetSession(*owner_view);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(owner_view->valid());
    EXPECT_TRUE(other_view->valid());
    EXPECT_EQ(owner_view->current_pos(), 2U);
    EXPECT_EQ(other_view->current_pos(), 3U);
    EXPECT_EQ(owner_view->token_capacity(), 6U);
    EXPECT_EQ(other_view->token_capacity(), 8U);
    EXPECT_FALSE(owner_view->awaiting_prefill());
    EXPECT_FALSE(other_view->awaiting_prefill());
    EXPECT_FALSE(owner.ReserveForSession(1, 1).ok());
    EXPECT_FALSE(other.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(owner.ReleaseSession(*owner_view).ok());
    EXPECT_TRUE(other.ReleaseSession(*other_view).ok());
}

TEST(KVCacheManager, ForeignResetDiagnosticFailurePreservesBothReservations) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    KVCacheManager owner;
    KVCacheManager other;
    ASSERT_TRUE(owner.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(other.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto owner_view = owner.ReserveForSession(2, 4);
    auto other_view = other.ReserveForSession(3, 5);
    ASSERT_TRUE(owner_view.ok());
    ASSERT_TRUE(other_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*owner_view, 2).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*other_view, 3).ok());

    test::ScopedMallocFailure failure(0);
    const Status status = other.ResetSession(*owner_view);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    EXPECT_TRUE(owner_view->valid());
    EXPECT_TRUE(other_view->valid());
    EXPECT_EQ(owner_view->current_pos(), 2U);
    EXPECT_EQ(other_view->current_pos(), 3U);
    EXPECT_EQ(owner_view->token_capacity(), 6U);
    EXPECT_EQ(other_view->token_capacity(), 8U);
    EXPECT_FALSE(owner_view->awaiting_prefill());
    EXPECT_FALSE(other_view->awaiting_prefill());
    EXPECT_FALSE(owner.ReserveForSession(1, 1).ok());
    EXPECT_FALSE(other.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(owner.ReleaseSession(*owner_view).ok());
    EXPECT_TRUE(other.ReleaseSession(*other_view).ok());
}

TEST(KVCacheManager, ReleaseRejectsForeignViewWithoutChangingEitherReservation) {
    KVCacheManager owner;
    KVCacheManager other;
    ASSERT_TRUE(owner.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(other.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto owner_view = owner.ReserveForSession(2, 4);
    auto other_view = other.ReserveForSession(3, 5);
    ASSERT_TRUE(owner_view.ok());
    ASSERT_TRUE(other_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*owner_view, 2).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*other_view, 3).ok());

    const Status status = other.ReleaseSession(*owner_view);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(owner_view->valid());
    EXPECT_TRUE(other_view->valid());
    EXPECT_EQ(owner_view->current_pos(), 2U);
    EXPECT_EQ(other_view->current_pos(), 3U);
    EXPECT_EQ(owner_view->token_capacity(), 6U);
    EXPECT_EQ(other_view->token_capacity(), 8U);
    EXPECT_FALSE(owner_view->awaiting_prefill());
    EXPECT_FALSE(other_view->awaiting_prefill());
    EXPECT_FALSE(owner.ReserveForSession(1, 1).ok());
    EXPECT_FALSE(other.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(owner.ReleaseSession(*owner_view).ok());
    EXPECT_TRUE(other.ReleaseSession(*other_view).ok());
}

TEST(KVCacheManager, ForeignReleaseDiagnosticFailurePreservesBothReservations) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    KVCacheManager owner;
    KVCacheManager other;
    ASSERT_TRUE(owner.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    ASSERT_TRUE(other.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto owner_view = owner.ReserveForSession(2, 4);
    auto other_view = other.ReserveForSession(3, 5);
    ASSERT_TRUE(owner_view.ok());
    ASSERT_TRUE(other_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*owner_view, 2).ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*other_view, 3).ok());

    test::ScopedMallocFailure failure(0);
    const Status status = other.ReleaseSession(*owner_view);
    const auto result = failure.Stop();

    EXPECT_TRUE(result.allocation_failed);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(status.message().empty());
    EXPECT_TRUE(owner_view->valid());
    EXPECT_TRUE(other_view->valid());
    EXPECT_EQ(owner_view->current_pos(), 2U);
    EXPECT_EQ(other_view->current_pos(), 3U);
    EXPECT_EQ(owner_view->token_capacity(), 6U);
    EXPECT_EQ(other_view->token_capacity(), 8U);
    EXPECT_FALSE(owner_view->awaiting_prefill());
    EXPECT_FALSE(other_view->awaiting_prefill());
    EXPECT_FALSE(owner.ReserveForSession(1, 1).ok());
    EXPECT_FALSE(other.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(owner.ReleaseSession(*owner_view).ok());
    EXPECT_TRUE(other.ReleaseSession(*other_view).ok());
}

TEST(KVCacheManager, ReinitializationPermanentlyInvalidatesActiveViewCopies) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto old_view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(old_view.ok());
    KVCacheSessionView old_copy = *old_view;
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*old_view, 3).ok());

    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 2, 24, 16, MakeKVType(), 64).ok());
    EXPECT_FALSE(old_view->valid());
    EXPECT_FALSE(old_copy.valid());
    auto fresh_view = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(fresh_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*fresh_view, 2).ok());

    EXPECT_FALSE(old_view->valid());
    EXPECT_FALSE(old_copy.valid());
    EXPECT_FALSE(old_copy.BindLayerForAppend(0, 0, 3).ok());
    EXPECT_FALSE(old_copy.KeyData(0, 0, 0).ok());
    EXPECT_FALSE(old_copy.ValueData(0, 0, 0).ok());
    EXPECT_EQ(manager.ResetSession(old_copy).code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(manager.ReleaseSession(old_copy).code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(fresh_view->valid());
    EXPECT_EQ(fresh_view->current_pos(), 2U);
    EXPECT_EQ(fresh_view->token_capacity(), 6U);
    EXPECT_FALSE(manager.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(manager.ReleaseSession(*fresh_view).ok());
}

TEST(KVCacheManager, FailedReinitializationPermanentlyInvalidatesActiveViewCopies) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto old_view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(old_view.ok());
    KVCacheSessionView old_copy = *old_view;
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*old_view, 3).ok());

    const Status status = manager.Init(test::TestCpuAllocator(), 1, 1, 0, 8, MakeKVType(), 64);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(manager.is_initialized());
    EXPECT_EQ(manager.total_bytes(), 0U);
    EXPECT_FALSE(old_view->valid());
    EXPECT_FALSE(old_copy.valid());
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto fresh_view = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(fresh_view.ok());
    ASSERT_TRUE(test::InitializeAndCommitKVForTest(*fresh_view, 2).ok());

    EXPECT_FALSE(old_view->valid());
    EXPECT_FALSE(old_copy.valid());
    EXPECT_EQ(manager.ResetSession(old_copy).code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(manager.ReleaseSession(old_copy).code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(fresh_view->valid());
    EXPECT_EQ(fresh_view->current_pos(), 2U);
    EXPECT_FALSE(manager.ReserveForSession(1, 1).ok());
    EXPECT_TRUE(manager.ReleaseSession(*fresh_view).ok());
}

TEST(KVCacheManager, ReleasedViewCopyRemainsStaleAfterReinitialization) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto old_view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(old_view.ok());
    KVCacheSessionView old_copy = *old_view;
    ASSERT_TRUE(manager.ReleaseSession(*old_view).ok());
    EXPECT_FALSE(old_copy.valid());

    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 16, 8, MakeKVType(), 64).ok());
    auto fresh_view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(fresh_view.ok());

    EXPECT_FALSE(old_copy.valid());
    EXPECT_TRUE(fresh_view->valid());
    EXPECT_EQ(manager.ReleaseSession(old_copy).code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(fresh_view->valid());
    EXPECT_TRUE(manager.ReleaseSession(*fresh_view).ok());
}

TEST(KVCacheManager, ReleaseInvalidatesViewAndAllowsNewReservation) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 2, 24, 8, MakeKVType(), 64).ok());
    StatusOr<KVCacheSessionView> view = manager.ReserveForSession(3, 5);
    ASSERT_TRUE(view.ok());

    ASSERT_TRUE(manager.ReleaseSession(*view).ok());
    EXPECT_FALSE(view->valid());
    EXPECT_FALSE(view->BindLayerForAppend(0, 0, 1).ok());

    StatusOr<KVCacheSessionView> next = manager.ReserveForSession(2, 4);
    ASSERT_TRUE(next.ok());
    EXPECT_TRUE(next->valid());
    EXPECT_EQ(next->current_pos(), 0U);
}

TEST(KVCacheManager, ReserveRejectsRequestsBeyondPhysicalCapacity) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 8, 8, MakeKVType(), 64).ok());

    const StatusOr<KVCacheSessionView> view = manager.ReserveForSession(6, 4);

    ASSERT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kOutOfRange);
}

TEST(KVCacheManager, ReserveRejectsEmptyPrompt) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 8, 8, MakeKVType(), 64).ok());

    const StatusOr<KVCacheSessionView> view = manager.ReserveForSession(0, 1);

    ASSERT_FALSE(view.ok());
    EXPECT_EQ(view.status().code(), StatusCode::kInvalidArgument);
}

TEST(KVCacheSessionView, AppendBindingRequiresTheCommittedFrontier) {
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(test::TestCpuAllocator(), 1, 1, 8, 8, MakeKVType(), 64).ok());
    const auto view = manager.ReserveForSession(2, 2);
    ASSERT_TRUE(view.ok());

    EXPECT_FALSE(view->BindLayerForAppend(0, 1, 2).ok());
    const auto append = view->BindLayerForAppend(0, 0, 2);
    ASSERT_TRUE(append.ok()) << append.status().ToString();
    EXPECT_EQ(append->begin, 0U);
    EXPECT_EQ(append->end, 2U);
    EXPECT_EQ(append->storage.layer_index, 0U);
    EXPECT_EQ(append->storage.token_capacity, 4U);
}

} // namespace
