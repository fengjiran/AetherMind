#include "aethermind/model/packed_weight_store.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/base/status.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>

namespace {

using namespace aethermind;

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

KernelSelector MakePackedCpuSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
}

class CountingPackedWeights final : public PackedWeights {
public:
    CountingPackedWeights(OpType op_type,
                          KernelSelector selector,
                          Buffer storage,
                          bool* destroyed_flag,
                          PackingRecipe recipe = {}) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          destroyed_flag_(destroyed_flag),
          recipe_(std::move(recipe)) {}

    ~CountingPackedWeights() override {
        if (destroyed_flag_ != nullptr) {
            *destroyed_flag_ = true;
        }
    }

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

    const PackingRecipe& recipe() const noexcept override {
        return recipe_;
    }

    DataType logical_dtype() const noexcept override {
        return {};
    }

    const std::vector<int64_t>& logical_shape() const noexcept override {
        return logical_shape_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
    bool* destroyed_flag_ = nullptr;
    PackingRecipe recipe_{};
    std::vector<int64_t> logical_shape_{};
};

TEST(PackedWeightStoreOwnership, StoreOwnsPackedWeightsUntilItIsDestroyed) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    {
        PackedWeightStore packed_weight_store;
        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(256),
                &destroyed);
        const PackedWeights* raw_ptr = packed.get();

        const WeightArtifactKey key{
                .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
                .selector = selector};
        ASSERT_TRUE(packed_weight_store.Store(
                                               key, std::shared_ptr<const PackedWeights>(std::move(packed)))
                            .ok());
        EXPECT_FALSE(destroyed);

        const auto found = packed_weight_store.Find(key);
        ASSERT_NE(found, nullptr);
        EXPECT_EQ(found.get(), raw_ptr);
        EXPECT_TRUE(found->storage().is_initialized());
    }

    EXPECT_TRUE(destroyed);
}

TEST(PackedWeightStoreOwnership, StoredPackedWeightsOutliveBackendInstance) {
    bool destroyed = false;
    const KernelSelector selector = MakePackedCpuSelector();

    PackedWeightStore packed_weight_store;
    {
        CpuBackend backend;
        (void) backend;

        auto packed = std::make_unique<CountingPackedWeights>(
                OpType::kLinear,
                selector,
                MakeTestBuffer(128),
                &destroyed);
        const WeightArtifactKey key{
                .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
                .selector = selector};
        ASSERT_TRUE(packed_weight_store.Store(
                                               key, std::shared_ptr<const PackedWeights>(std::move(packed)))
                            .ok());
    }

    const WeightArtifactKey key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};
    const auto found = packed_weight_store.Find(key);
    ASSERT_NE(found, nullptr);
    EXPECT_FALSE(destroyed);
    EXPECT_TRUE(found->storage().device().is_cpu());
}

TEST(PackedWeightStoreOwnership, StoreRejectsDuplicatePackedWeightEntries) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightArtifactKey key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};

    ASSERT_TRUE(packed_weight_store
                        .Store(key, std::make_shared<CountingPackedWeights>(
                                            OpType::kLinear, selector,
                                            MakeTestBuffer(64), nullptr))
                        .ok());

    const Status duplicate_status = packed_weight_store.Store(
            key, std::make_shared<CountingPackedWeights>(
                         OpType::kLinear, selector, MakeTestBuffer(64), nullptr));

    ASSERT_FALSE(duplicate_status.ok());
    EXPECT_EQ(duplicate_status.code(), StatusCode::kAlreadyExists);
}

TEST(PackedWeightStoreOwnership, DistinctRecipesCoexistForSameBindingAndSelector) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightBinding binding =
            MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ);
    const WeightArtifactKey base_key{.binding = binding, .selector = selector};
    const PackingRecipe recipe_a{.layout = "recipe_a", .alignment = 16};
    const PackingRecipe recipe_b{.layout = "recipe_b", .alignment = 32};

    // Two packing variants of the same logical weight coexist: the recipe
    // discriminates artifacts within one {binding, selector}.
    ASSERT_TRUE(packed_weight_store
                        .Store(WeightArtifactKey{.binding = binding,
                                                 .selector = selector,
                                                 .recipe = recipe_a},
                               std::make_shared<CountingPackedWeights>(
                                       OpType::kLinear, selector,
                                       MakeTestBuffer(16), nullptr, recipe_a))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(WeightArtifactKey{.binding = binding,
                                                 .selector = selector,
                                                 .recipe = recipe_b},
                               std::make_shared<CountingPackedWeights>(
                                       OpType::kLinear, selector,
                                       MakeTestBuffer(16), nullptr, recipe_b))
                        .ok());
    ASSERT_EQ(packed_weight_store.size(), 2U);
    ASSERT_NE(packed_weight_store.Find(
                      WeightArtifactKey{.binding = binding,
                                        .selector = selector,
                                        .recipe = recipe_a}),
              nullptr);
    ASSERT_NE(packed_weight_store.Find(
                      WeightArtifactKey{.binding = binding,
                                        .selector = selector,
                                        .recipe = recipe_b}),
              nullptr);

    // A {binding, selector} lookup is ambiguous across recipes and fails.
    const auto ambiguous =
            packed_weight_store.FindByBindingSelector(binding, selector);
    ASSERT_FALSE(ambiguous.ok());
    EXPECT_EQ(ambiguous.status().code(), StatusCode::kFailedPrecondition);
}

TEST(PackedWeightStoreOwnership, DistinctBindingsShareSelectorWithoutCollision) {
    PackedWeightStore packed_weight_store;
    const KernelSelector selector = MakePackedCpuSelector();
    const WeightArtifactKey q_key{
            .binding = MakeTransformerWeightBinding(0, TransformerWeightRole::kAttentionQ),
            .selector = selector};
    const WeightArtifactKey v_key{
            .binding = MakeTransformerWeightBinding(2, TransformerWeightRole::kAttentionV),
            .selector = selector};

    ASSERT_TRUE(packed_weight_store
                        .Store(q_key, std::make_shared<CountingPackedWeights>(
                                              OpType::kLinear, selector,
                                              MakeTestBuffer(64), nullptr))
                        .ok());
    ASSERT_TRUE(packed_weight_store
                        .Store(v_key, std::make_shared<CountingPackedWeights>(
                                              OpType::kLinear, selector,
                                              MakeTestBuffer(64), nullptr))
                        .ok());

    ASSERT_EQ(packed_weight_store.size(), 2U);
    ASSERT_NE(packed_weight_store.Find(q_key), nullptr);
    ASSERT_NE(packed_weight_store.Find(v_key), nullptr);
    EXPECT_NE(packed_weight_store.Find(q_key), packed_weight_store.Find(v_key));
}

}// namespace
