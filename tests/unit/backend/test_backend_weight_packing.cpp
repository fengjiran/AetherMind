#include "aethermind/backend/backend.h"
#include "aethermind/base/tensor_view.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace {

using namespace aethermind;

KernelSelector MakePackedSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

TensorView MakeWeightView() {
    static constexpr float weight = 1.0F;
    static constexpr std::array<int64_t, 2> shape{1, 1};
    static constexpr std::array<int64_t, 2> strides{1, 1};
    return TensorView(&weight, DataType::Float32(), shape, strides);
}

class QueryProbeBackend : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    StatusOr<ResolvedKernel> PrepareKernel(
            OpType, const KernelSelector&, const OpParams&) const override {
        return Status::NotFound("Test backend does not prepare kernels");
    }

    StatusOr<PackingRecipe> GetPackingRecipe(
            OpType, const KernelSelector&) const override {
        ++recipe_queries;
        return Status::NotFound("Test recipe lookup failed");
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

    mutable size_t recipe_queries = 0;
};

class RecordingPackingBackend final : public QueryProbeBackend {
public:
    StatusOr<std::unique_ptr<PackedWeight>> PackWeights(
            OpType op_type,
            std::span<const TensorView> components,
            const KernelSelector& selector,
            const PackingRecipe& recipe) const override {
        ++pack_calls;
        received_op_type = op_type;
        received_components = components.data();
        received_component_count = components.size();
        received_selector = selector;
        received_recipe = recipe;
        return Status::ResourceExhausted("Test packing failure");
    }

    mutable size_t pack_calls = 0;
    mutable OpType received_op_type = OpType::kUnknown;
    mutable const TensorView* received_components = nullptr;
    mutable size_t received_component_count = 0;
    mutable KernelSelector received_selector{};
    mutable PackingRecipe received_recipe{};
};

TEST(BackendWeightPacking, DefaultExplicitPackingReturnsUnimplementedWithoutRecipeLookup) {
    QueryProbeBackend backend;
    const std::array<TensorView, 1> components{MakeWeightView()};
    const PackingRecipe recipe{.layout = PackingLayout::kCpuIdentity, .alignment = 64};

    const auto packed = backend.PackWeights(
            OpType::kLinear, components, MakePackedSelector(), recipe);

    EXPECT_EQ(packed.status().code(), StatusCode::kUnimplemented);
    EXPECT_EQ(backend.recipe_queries, 0U);
}

TEST(BackendWeightPacking, ExplicitPackingForwardsRecipeAndInputsToVirtualImplementation) {
    RecordingPackingBackend backend;
    const Backend& abstract_backend = backend;
    const std::array<TensorView, 1> components{MakeWeightView()};
    const KernelSelector selector = MakePackedSelector();
    const PackingRecipe recipe{.layout = PackingLayout::kCpuBPanelF32Kc512Nr16, .alignment = 64};

    const auto packed = abstract_backend.PackWeights(OpType::kLinear, components, selector, recipe);

    EXPECT_EQ(packed.status().code(), StatusCode::kResourceExhausted);
    EXPECT_EQ(packed.status().message(), "Test packing failure");
    EXPECT_EQ(backend.recipe_queries, 0U);
    EXPECT_EQ(backend.pack_calls, 1U);
    EXPECT_EQ(backend.received_op_type, OpType::kLinear);
    EXPECT_EQ(backend.received_components, components.data());
    EXPECT_EQ(backend.received_component_count, components.size());
    EXPECT_EQ(backend.received_selector, selector);
    EXPECT_EQ(backend.received_recipe, recipe);
}

} // namespace
