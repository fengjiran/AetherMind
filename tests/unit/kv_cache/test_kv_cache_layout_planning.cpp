#include "aethermind/kv_cache/kv_cache_layout.h"
#include "aethermind/kv_cache/kv_cache_manager.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/memory/cpu_allocator.h"
#include "inference/test_malloc_interposer.h"

#include <array>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

constexpr size_t kMaxSize = std::numeric_limits<size_t>::max();

struct GeometryCase {
    const char* name;
    size_t layers;
    size_t heads;
    size_t tokens;
    size_t head_dim;
    DataType dtype;
    size_t alignment;
    size_t padded_head_dim;
    size_t token_stride;
    size_t head_stride;
    size_t layer_stride;
    size_t plane_bytes;
    size_t last_offset;
};

const std::array kGeometryCases{
        GeometryCase{"F32Tail", 2, 3, 4, 17, DataType::Float32(), 64, 32, 128, 512, 1536, 3072, 3008},
        GeometryCase{"F16MultiHead", 2, 4, 8, 17, DataType::Float(16), 64, 32, 64, 512, 2048, 4096, 4064},
        GeometryCase{"BF16SingleHead", 3, 1, 7, 33, DataType::BFloat(16), 16, 40, 80, 560, 560, 1680, 1664},
        GeometryCase{"AlignmentBelowElement", 1, 1, 2, 3, DataType::Float32(), 1, 3, 12, 24, 24, 24, 20},
        GeometryCase{"LargeAlignment", 1, 2, 3, 65, DataType::Float32(), 256, 128, 512, 1536, 3072, 3072, 2816},
        GeometryCase{"ThreeByteElement", 1, 2, 2, 17, DataType::UInt(8, 3), 64, 64, 192, 384, 768, 768, 624},
        GeometryCase{"TwelveByteElement", 2, 2, 3, 5, DataType::Float(32, 3), 16, 8, 96, 288, 576, 1152, 1104},
        GeometryCase{"ElementLargerThanAlignment", 1, 1, 2, 3, DataType::Float(32, 3), 8, 4, 48, 96, 96, 96, 72},
};

void ExpectLayoutMatches(const KVCacheLayout& layout, const GeometryCase& expected) {
    EXPECT_EQ(layout.num_layers, expected.layers);
    EXPECT_EQ(layout.num_kv_heads, expected.heads);
    EXPECT_EQ(layout.max_tokens, expected.tokens);
    EXPECT_EQ(layout.head_dim, expected.head_dim);
    EXPECT_EQ(layout.kv_dtype, expected.dtype);
    EXPECT_EQ(layout.alignment, expected.alignment);
    EXPECT_EQ(layout.padded_head_dim, expected.padded_head_dim);
    EXPECT_EQ(layout.token_stride, expected.token_stride);
    EXPECT_EQ(layout.head_stride, expected.head_stride);
    EXPECT_EQ(layout.layer_stride, expected.layer_stride);
    EXPECT_TRUE(layout.Validate().ok());

    const auto plane = layout.BytesPerPlane();
    ASSERT_TRUE(plane.ok());
    EXPECT_EQ(*plane, expected.plane_bytes);
    const auto total = layout.TotalBytes();
    ASSERT_TRUE(total.ok());
    EXPECT_EQ(*total, 2 * expected.plane_bytes);
    const auto last = layout.Offset(expected.layers - 1, expected.heads - 1,
                                    expected.tokens - 1, expected.head_dim - 1);
    ASSERT_TRUE(last.ok());
    EXPECT_EQ(*last, expected.last_offset);
    EXPECT_LE(*last + layout.ElementBytes(), *plane);
}

struct AllocationStats {
    std::vector<AllocationRequest> requests;
    size_t live_buffers = 0;
    size_t released_buffers = 0;
};

struct TrackedBuffer {
    Buffer backing;
    std::shared_ptr<AllocationStats> stats;

    ~TrackedBuffer() {
        --stats->live_buffers;
        ++stats->released_buffers;
    }
};

void ReleaseTrackedBuffer(void* context, void*) noexcept {
    delete static_cast<TrackedBuffer*>(context);
}

class RecordingAllocator final : public Allocator {
public:
    explicit RecordingAllocator(std::shared_ptr<AllocationStats> stats)
        : stats_(std::move(stats)), backing_(Device::CPU()) {}

    Device device() const noexcept override {
        return Device::CPU();
    }

    size_t fail_on_call = kMaxSize;

protected:
    StatusOr<MemoryHandle> AllocateStorage(const AllocationRequest& request) override {
        const size_t call = stats_->requests.size();
        stats_->requests.push_back(request);
        if (call == fail_on_call) {
            return Status::ResourceExhausted({});
        }

        auto buffer = backing_.TryAllocate(request);
        if (!buffer.ok()) {
            return buffer.status();
        }

        // The release context owns the backing Buffer and the observation
        // state independently of this allocator's lifetime.
        auto owner = std::make_unique<TrackedBuffer>(std::move(*buffer), stats_);
        void* data = owner->backing.mutable_data();
        const size_t alignment = owner->backing.alignment();
        ++stats_->live_buffers;
        return MemoryHandle(data, owner.release(), &ReleaseTrackedBuffer,
                            Device::CPU(), alignment);
    }

private:
    std::shared_ptr<AllocationStats> stats_;
    CPUAllocator backing_;
};

TEST(KVCacheLayoutPlanning, ComputesPhysicalGeometryAndOffsets) {
    for (const auto& expected: kGeometryCases) {
        SCOPED_TRACE(expected.name);
        const auto planned =
                PlanKVCacheLayout(expected.layers, expected.heads,
                                  expected.tokens, expected.head_dim,
                                  expected.dtype, expected.alignment);
        ASSERT_TRUE(planned.ok()) << planned.status().ToString();
        ExpectLayoutMatches(*planned, expected);
    }
}

TEST(KVCacheLayoutPlanning, RejectsZeroDimensions) {
    for (size_t zero_index = 0; zero_index < 4; ++zero_index) {
        SCOPED_TRACE(zero_index);
        std::array<size_t, 4> dimensions{1, 1, 1, 1};
        dimensions[zero_index] = 0;
        const auto planned =
                PlanKVCacheLayout(dimensions[0], dimensions[1],
                                  dimensions[2], dimensions[3],
                                  DataType::Float32(), 64);
        EXPECT_EQ(planned.status().code(), StatusCode::kInvalidArgument);
    }
}

TEST(KVCacheLayoutPlanning, RejectsInvalidAlignment) {
    for (const size_t alignment: {size_t{0}, size_t{3}, size_t{24}}) {
        SCOPED_TRACE(alignment);
        const auto planned =
                PlanKVCacheLayout(1, 1, 1, 1,
                                  DataType::Float32(), alignment);
        EXPECT_EQ(planned.status().code(), StatusCode::kInvalidArgument);
    }
}

TEST(KVCacheLayoutPlanning, RejectsDTypesWithoutFixedPositiveLanes) {
    const auto valid =
            PlanKVCacheLayout(1, 1, 1, 1, DataType::Float32(), 64);
    ASSERT_TRUE(valid.ok());
    for (const auto dtype: {DataType{}, DataType::ScalableFloat(32, 2),
                            DataType::Float(32, 65535)}) {
        const auto planned = PlanKVCacheLayout(1, 1, 1, 1, dtype, 64);
        EXPECT_EQ(planned.status().code(), StatusCode::kInvalidArgument);
        auto invalid = *valid;
        invalid.kv_dtype = dtype;
        EXPECT_EQ(invalid.ElementBytes(), 0U);
        EXPECT_EQ(invalid.Validate().code(), StatusCode::kInvalidArgument);
    }
}

TEST(KVCacheLayoutPlanning, RejectsOverflowAtEveryLayoutStage) {
    struct OverflowCase {
        const char* name;
        size_t layers = 1;
        size_t heads = 1;
        size_t tokens = 1;
        size_t head_dim = 1;
        DataType dtype = DataType::Float32();
        size_t alignment = 64;
    };

    const std::array cases{
            OverflowCase{.name = "Padding", .head_dim = kMaxSize, .dtype = DataType::UInt(8)},
            OverflowCase{.name = "TokenStride", .head_dim = kMaxSize / 4 + 1, .alignment = 1},
            OverflowCase{.name = "HeadStride", .tokens = kMaxSize / 64 + 1},
            OverflowCase{.name = "LayerStride", .heads = kMaxSize / 64 + 1},
            OverflowCase{.name = "PlaneBytes", .layers = kMaxSize / 64 + 1},
            OverflowCase{.name = "TotalBytes", .layers = kMaxSize / 128 + 1},
            OverflowCase{.name = "RowElementGranularity", .dtype = DataType::Float(32, 3), .alignment = size_t{1} << (std::numeric_limits<size_t>::digits - 1)},
    };

    for (const auto& expected: cases) {
        SCOPED_TRACE(expected.name);
        const auto planned = PlanKVCacheLayout(expected.layers, expected.heads,
                                               expected.tokens, expected.head_dim,
                                               expected.dtype, expected.alignment);
        EXPECT_EQ(planned.status().code(), StatusCode::kOverflow);
    }
}

TEST(KVCacheLayoutPlanning, AcceptsRepresentableTwoPlaneBoundaryWithoutAllocatingStorage) {
    const auto planned =
            PlanKVCacheLayout(1, 1, 1, kMaxSize / 2,
                              DataType::UInt(8), 1);
    ASSERT_TRUE(planned.ok()) << planned.status().ToString();
    const auto total = planned->TotalBytes();
    ASSERT_TRUE(total.ok());
    EXPECT_EQ(*total, kMaxSize - 1);
}

TEST(KVCacheLayoutPlanning, TotalBytesIncludesDeclaredStrideGaps) {
    const KVCacheLayout layout{
            .num_layers = 2,
            .num_kv_heads = 3,
            .max_tokens = 4,
            .head_dim = 5,
            .padded_head_dim = 8,
            .token_stride = 32,
            .head_stride = 160,
            .layer_stride = 640,
            .kv_dtype = DataType::Float(16),
            .alignment = 16,
    };
    const auto total = layout.TotalBytes();
    ASSERT_TRUE(total.ok());
    EXPECT_EQ(*total, 2560U);
}

TEST(KVCacheLayoutPlanning, TotalBytesRejectsInvalidLayoutsAndPlaneOverflow) {
    const auto planned = PlanKVCacheLayout(1, 1, 1, 1, DataType::Float32(), 64);
    ASSERT_TRUE(planned.ok());
    auto invalid = *planned;
    invalid.token_stride = 15;
    EXPECT_EQ(invalid.TotalBytes().status().code(), StatusCode::kInvalidArgument);
    invalid = *planned;
    invalid.kv_dtype = DataType::ScalableFloat(32, 2);
    EXPECT_EQ(invalid.ElementBytes(), 0U);
    EXPECT_EQ(invalid.Validate().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(invalid.TotalBytes().status().code(), StatusCode::kInvalidArgument);
    auto plane_overflow = *planned;
    plane_overflow.num_layers = kMaxSize / 64 + 1;
    EXPECT_EQ(plane_overflow.TotalBytes().status().code(), StatusCode::kOverflow);
    auto total_overflow = *planned;
    total_overflow.num_layers = kMaxSize / 128 + 1;
    ASSERT_TRUE(total_overflow.BytesPerPlane().ok());
    EXPECT_EQ(total_overflow.TotalBytes().status().code(), StatusCode::kOverflow);
}

TEST(KVCacheLayoutPlanning, ManagerAllocatesExactlyThePlannedPlanes) {
    for (const auto& expected: kGeometryCases) {
        SCOPED_TRACE(expected.name);
        const auto stats = std::make_shared<AllocationStats>();
        RecordingAllocator allocator(stats);
        {
            KVCacheManager manager;
            const auto status = manager.Init(allocator, expected.layers, expected.heads,
                                             expected.tokens, expected.head_dim,
                                             expected.dtype, expected.alignment);
            ASSERT_TRUE(status.ok()) << status.ToString();
            ExpectLayoutMatches(manager.layout(), expected);
            ASSERT_EQ(stats->requests.size(), 2U);
            for (const auto& request: stats->requests) {
                EXPECT_EQ(request.nbytes, expected.plane_bytes);
                EXPECT_EQ(request.alignment, expected.alignment);
            }
            EXPECT_EQ(manager.total_bytes(), 2 * expected.plane_bytes);
            EXPECT_EQ(stats->live_buffers, 2U);
        }
        EXPECT_EQ(stats->live_buffers, 0U);
        EXPECT_EQ(stats->released_buffers, 2U);
    }
}

TEST(KVCacheLayoutPlanning, SecondPlaneFailureReleasesStorageAndCanRetry) {
    const auto stats = std::make_shared<AllocationStats>();
    RecordingAllocator allocator(stats);
    allocator.fail_on_call = 1;
    KVCacheManager manager;
    const auto status = manager.Init(allocator, 2, 3, 4, 17, DataType::Float32(), 64);
    EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
    EXPECT_FALSE(manager.is_initialized());
    EXPECT_EQ(manager.total_bytes(), 0U);
    EXPECT_EQ(manager.capacity_tokens(), 0U);
    EXPECT_EQ(stats->requests.size(), 2U);
    EXPECT_EQ(stats->live_buffers, 0U);
    EXPECT_EQ(stats->released_buffers, 1U);

    allocator.fail_on_call = kMaxSize;
    ASSERT_TRUE(manager.Init(allocator, 2, 3, 4, 17, DataType::Float32(), 64).ok());
    EXPECT_EQ(stats->requests.size(), 4U);
    EXPECT_EQ(stats->live_buffers, 2U);
    auto view = manager.ReserveForSession(2, 1);
    ASSERT_TRUE(view.ok());
    EXPECT_TRUE(view->valid());
    EXPECT_TRUE(manager.ReleaseSession(*view).ok());
}

TEST(KVCacheLayoutPlanning, PlanningFailureClearsOldStorageBeforeReturning) {
    const auto stats = std::make_shared<AllocationStats>();
    RecordingAllocator allocator(stats);
    KVCacheManager manager;
    ASSERT_TRUE(manager.Init(allocator, 1, 1, 8, 8, DataType::Float32(), 64).ok());
    auto view = manager.ReserveForSession(2, 2);
    ASSERT_TRUE(view.ok());
    const KVCacheSessionView old = *view;

    const auto status = manager.Init(allocator, 1, 1, 1, kMaxSize,
                                     DataType::UInt(8), 64);
    EXPECT_EQ(status.code(), StatusCode::kOverflow);
    EXPECT_FALSE(manager.is_initialized());
    EXPECT_EQ(manager.total_bytes(), 0U);
    EXPECT_EQ(manager.capacity_tokens(), 0U);
    EXPECT_EQ(stats->requests.size(), 2U);
    EXPECT_EQ(stats->released_buffers, 2U);
    EXPECT_EQ(stats->live_buffers, 0U);
    EXPECT_FALSE(old.valid());
    EXPECT_FALSE(view->valid());

    ASSERT_TRUE(manager.Init(allocator, 1, 1, 8, 8, DataType::Float32(), 64).ok());
    auto fresh = manager.ReserveForSession(2, 2);
    ASSERT_TRUE(fresh.ok());
    EXPECT_TRUE(fresh->valid());
    EXPECT_FALSE(old.valid());
    EXPECT_EQ(manager.ResetSession(old).code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(manager.ReleaseSession(*fresh).ok());
}

TEST(KVCacheLayoutPlanningNoAllocation, SucceedsWithoutHeapAllocation) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    test::BeginMallocCallCounting();
    const auto planned = PlanKVCacheLayout(2, 3, 4, 17, DataType::Float32(), 64);
    const StatusOr<size_t> total = planned.ok()
                                           ? planned->TotalBytes()
                                           : StatusOr<size_t>(planned.status());
    const auto counts = test::EndMallocCallCounting();

    ASSERT_TRUE(planned.ok());
    ASSERT_TRUE(total.ok());
    EXPECT_EQ(*total, 6144U);
    EXPECT_EQ(counts.malloc_calls, 0U);
    EXPECT_EQ(counts.calloc_calls, 0U);
    EXPECT_EQ(counts.realloc_calls, 0U);
    EXPECT_EQ(counts.free_calls, 0U);
    EXPECT_EQ(counts.aligned_alloc_calls, 0U);
    EXPECT_EQ(counts.posix_memalign_calls, 0U);
    EXPECT_EQ(counts.memalign_calls, 0U);
}

TEST(KVCacheLayoutPlanningDiagnosticFailure, ReturnsResourceExhaustedWithoutPublishingLayout) {
    if (!test::MallocInterposerAvailable()) {
        GTEST_SKIP() << "Requires the glibc malloc interposer";
    }
    const auto valid = PlanKVCacheLayout(1, 1, 1, 1, DataType::Float32(), 64);
    ASSERT_TRUE(valid.ok());
    auto total_overflow = *valid;
    total_overflow.num_layers = kMaxSize / 128 + 1;
    for (const bool planner_failure: {true, false}) {
        SCOPED_TRACE(planner_failure);
        test::ScopedMallocFailure failure(0);
        const Status status = planner_failure
                                      ? PlanKVCacheLayout(1, 1, 1, 1, DataType::Float32(), 3).status()
                                      : total_overflow.TotalBytes().status();
        const auto observed = failure.Stop();

        EXPECT_TRUE(observed.allocation_failed);
        EXPECT_EQ(status.code(), StatusCode::kResourceExhausted);
        EXPECT_TRUE(status.message().empty());
    }
}

} // namespace
