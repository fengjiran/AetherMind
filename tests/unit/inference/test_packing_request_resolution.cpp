#include "aethermind/backend/backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/operators/op_params.h"
#include "inference/executable_model_internal.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

class RecipeLookupTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType,
            const KernelSelector&,
            const OpParams&) const override {
        return Status::NotFound("RecipeLookupTestBackend does not resolve kernels");
    }

    StatusOr<PackingRecipe> GetPackingRecipe(
            OpType op_type,
            const KernelSelector& selector) const override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return Status::InvalidArgument("test expected a packed selector");
        }
        if (op_type == OpType::kLinear) {
            return CpuIdentityPackingRecipe();
        }
        if (op_type == OpType::kQkvLinear) {
            return cpu::CpuBPanelF32V1Avx2Recipe();
        }
        return Status::NotFound("test recipe is not registered for this op");
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

KernelSelector MakePackedSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

WeightPackingRequest MakeRequest(uint32_t value_index, OpType op_type) {
    return WeightPackingRequest{
            .op_type = op_type,
            .source_id = 13,
            .value_index = value_index,
            .binding = {},
            .selector = MakePackedSelector(),
    };
}

TEST(PackingRequestResolution, CoalescesConsumersWithSameRecipe) {
    RecipeLookupTestBackend backend;
    auto resolved = inference::internal::ResolveWeightPackingRequests(
            backend,
            {MakeRequest(4, OpType::kLinear), MakeRequest(4, OpType::kLinear)},
            DeviceType::kCPU);
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    ASSERT_EQ(resolved->size(), 1U);
    EXPECT_EQ(resolved->front().recipe, CpuIdentityPackingRecipe());
}

TEST(PackingRequestResolution, RejectsSharedValueWithIncompatibleRecipes) {
    RecipeLookupTestBackend backend;
    auto resolved = inference::internal::ResolveWeightPackingRequests(
            backend,
            {MakeRequest(4, OpType::kLinear), MakeRequest(4, OpType::kQkvLinear)},
            DeviceType::kCPU);
    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
    EXPECT_NE(resolved.status().message().find("incompatible op types or packing recipes"),
              std::string::npos);
}

TEST(PackingRequestResolution, KeepsDistinctWeightValuesSeparate) {
    RecipeLookupTestBackend backend;
    auto resolved = inference::internal::ResolveWeightPackingRequests(
            backend,
            {MakeRequest(4, OpType::kLinear), MakeRequest(5, OpType::kQkvLinear)},
            DeviceType::kCPU);
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    ASSERT_EQ(resolved->size(), 2U);
    EXPECT_EQ((*resolved)[0].recipe, CpuIdentityPackingRecipe());
    EXPECT_EQ((*resolved)[1].recipe, cpu::CpuBPanelF32V1Avx2Recipe());
}

} // namespace
