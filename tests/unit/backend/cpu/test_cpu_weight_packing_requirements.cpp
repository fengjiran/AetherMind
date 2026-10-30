#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/model/weight/packed_weight_collection.h"
#include "aethermind/model/weight/weight_packing_request.h"
#include "backend/cpu/cpu_backend_internal.h"
#include "test_utils/test_resource_allocator.h"

#include <array>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;
KernelSelector Selector() { return {.device_type = DeviceType::kCPU, .act_dtype = DataType::Float32(), .weight_dtype = DataType::Float32(), .weight_format = WeightFormat::kPacked, .phase = ExecPhase::kBoth}; }
struct PackingStorage final : RawStorage {
    std::array<float, 15> data{};
};
WeightPackingRequest Request(const std::shared_ptr<PackingStorage>& owner) {
    return {.op_type = OpType::kLinear, .source_id = 7, .value_index = 1, .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ), .components = {{.data = reinterpret_cast<const std::byte*>(owner->data.data()), .bytes = 60, .dtype = DataType::Float32(), .shape = {3, 5}, .storage = owner}}, .selector = Selector(), .recipe = CpuIdentityPackingRecipe()};
}
class LegacyPackingBackend : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType, const KernelSelector&, const OpParams&) const override { return Status::Unimplemented({}); }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }
    StatusOr<std::unique_ptr<PackedWeight>> PackWeights(OpType op, std::span<const TensorView> values,
                                                        const KernelSelector& selector, const PackingRecipe& recipe, Allocator& allocator) const override {
        return cpu::internal::PackWeightsWithRecipe(op, values, selector, recipe, allocator);
    }
};
class PlanningPackingBackend final : public LegacyPackingBackend {
public:
    bool mismatch = false;
    StatusOr<AllocationRequest> GetWeightPackingRequirements(OpType op, std::span<const TensorView> values,
                                                             const KernelSelector& selector, const PackingRecipe& recipe) const override {
        auto result = cpu::internal::GetWeightPackingRequirementsWithRecipe(op, values, selector, recipe);
        if (result.ok() && mismatch) result->nbytes += 4;
        return result;
    }
};
TEST(CpuWeightPackingRequirements, IdentityAndPanelMatchActualStorageRequests) {
    const std::array<int64_t, 2> shape{3, 5}, stride{5, 1};
    std::array<float, 15> data{};
    const std::array<TensorView, 1> values{TensorView(data.data(), DataType::Float32(), shape, stride, 4)};
    for (const auto recipe: {CpuIdentityPackingRecipe(), cpu::CpuBPanelF32Kc512Nr16Recipe()}) {
        auto requirement = cpu::internal::GetWeightPackingRequirementsWithRecipe(OpType::kLinear, values, Selector(), recipe);
        ASSERT_TRUE(requirement.ok());
        auto stats = std::make_shared<test::ResourceAllocationStats>();
        test::ResourceRecordingAllocator allocator(stats);
        auto packed = cpu::internal::PackWeightsWithRecipe(OpType::kLinear, values, Selector(), recipe, allocator);
        ASSERT_TRUE(packed.ok());
        ASSERT_EQ(stats->requests.size(), 1U);
        EXPECT_EQ(stats->requests[0].nbytes, requirement->nbytes);
        EXPECT_EQ((*packed)->storage().nbytes(), requirement->nbytes);
        EXPECT_EQ(requirement->nbytes, recipe == CpuIdentityPackingRecipe() ? 60U : 32768U);
    }
}
TEST(CpuWeightPackingRequirements, QueryDoesNotReadPayloadAndRejectsInvalidMetadata) {
    const std::array<int64_t, 2> shape{3, 5}, stride{5, 1};
    const std::array<TensorView, 1> values{TensorView(reinterpret_cast<const void*>(uintptr_t{64}), DataType::Float32(), shape, stride, 4)};
    for (const auto recipe: {CpuIdentityPackingRecipe(), cpu::CpuBPanelF32Kc512Nr16Recipe()}) {
        EXPECT_TRUE(cpu::internal::GetWeightPackingRequirementsWithRecipe(OpType::kLinear, values, Selector(), recipe).ok());
    }
    EXPECT_EQ(cpu::internal::GetWeightPackingRequirementsWithRecipe(OpType::kLinear, values, Selector(), {}).status().code(), StatusCode::kInvalidArgument);
    const std::array<int64_t, 2> huge{std::numeric_limits<int64_t>::max(), 2}, huge_stride{2, 1};
    const std::array<TensorView, 1> overflow{TensorView(reinterpret_cast<const void*>(uintptr_t{64}), DataType::Float32(), huge, huge_stride, 4)};
    EXPECT_EQ(cpu::internal::GetWeightPackingRequirementsWithRecipe(OpType::kLinear, overflow, Selector(), CpuIdentityPackingRecipe()).status().code(), StatusCode::kOverflow);
}
TEST(PackingResourceLimits, ChecksWholeBatchBeforeAnyAllocation) {
    auto owner = std::make_shared<PackingStorage>();
    auto first = Request(owner), second = first;
    second.value_index = 2;
    PlanningPackingBackend backend;
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    EXPECT_EQ(PrepackWeightRequests(backend, {first, second}, allocator, 119).status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());
    auto packed = PrepackWeightRequests(backend, {first, second}, allocator, 120);
    ASSERT_TRUE(packed.ok());
    auto bytes = packed->total_storage_bytes();
    ASSERT_TRUE(bytes.ok());
    EXPECT_EQ(*bytes, 120U);
    EXPECT_TRUE(packed->storage_plan_complete());
}
TEST(PackingResourceLimits, UnknownQueryOnlyPermittedForUnlimitedBatch) {
    LegacyPackingBackend backend;
    auto owner = std::make_shared<PackingStorage>();
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    EXPECT_EQ(PrepackWeightRequests(backend, {Request(owner)}, allocator, 100).status().code(), StatusCode::kUnimplemented);
    EXPECT_TRUE(stats->requests.empty());
    auto packed = PrepackWeightRequests(backend, {Request(owner)}, allocator);
    ASSERT_TRUE(packed.ok());
    EXPECT_FALSE(packed->storage_plan_complete());
}
TEST(PackingResourceLimits, RejectsQueryAndActualStorageMismatch) {
    PlanningPackingBackend backend;
    backend.mismatch = true;
    auto owner = std::make_shared<PackingStorage>();
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    EXPECT_EQ(PrepackWeightRequests(backend, {Request(owner)}, allocator, 100).status().code(), StatusCode::kInternal);
}
TEST(PackingResourceLimits, CountsSharedBufferImplementationOnlyOnce) {
    PlanningPackingBackend backend;
    auto owner = std::make_shared<PackingStorage>();
    auto request = Request(owner);
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    auto source = PrepackWeightRequests(backend, {request}, allocator);
    ASSERT_TRUE(source.ok());
    WeightArtifactKey key{.source_id = request.source_id, .value_index = request.value_index, .binding = request.binding, .selector = request.selector, .recipe = request.recipe};
    auto artifact = source->Find(key);
    ASSERT_NE(artifact, nullptr);
    PackedWeightCollection aliases;
    EXPECT_TRUE(aliases.Insert(key, artifact).ok());
    key.value_index = 2;
    EXPECT_TRUE(aliases.Insert(key, artifact).ok());
    auto bytes = aliases.total_storage_bytes();
    ASSERT_TRUE(bytes.ok());
    EXPECT_EQ(*bytes, 60U);
}
} // namespace
