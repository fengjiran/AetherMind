#include "aethermind/model/weight_prepack_planner.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/model/model_instance_builder.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <vector>

namespace aethermind {
namespace {

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

HfModelConfig MakeLlamaConfig(int64_t num_layers) {
    return HfModelConfig{
            .model_type = "llama",
            .architectures = {"LlamaForCausalLM"},
            .hidden_size = 64,
            .intermediate_size = 256,
            .num_hidden_layers = num_layers,
            .num_attention_heads = 8,
            .num_key_value_heads = 4,
            .vocab_size = 1000,
            .rms_norm_eps = 1e-6,
            .tie_word_embeddings = false,
    };
}

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage,
                             size_t offset,
                             size_t nbytes,
                             DataType dtype,
                             const std::vector<int64_t>& shape) {
    return RawWeightView{
            .data = storage->data.data() + offset,
            .bytes = nbytes,
            .dtype = dtype,
            .shape = shape,
            .storage = storage,
    };
}

DecoderLayerRawWeights MakeTestLayer(const std::shared_ptr<TestStorage>& storage,
                                     size_t base_offset) {
    DecoderLayerRawWeights layer;
    layer.attn.q_proj = MakeWeightView(storage, base_offset + 0, 8, DataType::Float32(), {2, 1});
    layer.attn.k_proj = MakeWeightView(storage, base_offset + 8, 8, DataType::Float32(), {2, 1});
    layer.attn.v_proj = MakeWeightView(storage, base_offset + 16, 8, DataType::Float32(), {2, 1});
    layer.attn.o_proj = MakeWeightView(storage, base_offset + 24, 8, DataType::Float32(), {2, 1});
    layer.ffn.gate_proj = MakeWeightView(storage, base_offset + 32, 8, DataType::Float32(), {2, 1});
    layer.ffn.up_proj = MakeWeightView(storage, base_offset + 40, 8, DataType::Float32(), {2, 1});
    layer.ffn.down_proj = MakeWeightView(storage, base_offset + 48, 8, DataType::Float32(), {2, 1});
    layer.norm.input_rmsnorm = MakeWeightView(storage, base_offset + 56, 8, DataType::Float32(), {2, 1});
    layer.norm.post_attn_rmsnorm = MakeWeightView(storage, base_offset + 64, 8, DataType::Float32(), {2, 1});
    return layer;
}

KernelSelector MakeExpectedSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kAVX2,
            .phase = ExecPhase::kBoth,
    };
}

TEST(WeightPrepackPlannerTest, BuildRequestsEnumeratesAllLinearWeightsPerLayer) {
    auto storage = std::make_shared<TestStorage>(256);

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));
    index.layers.push_back(MakeTestLayer(storage, 100));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(2), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    // 2 layers × 7 linear weights = 14 requests.
    EXPECT_EQ(requests->size(), 14);

    // All requests should be kLinear with the expected packed selector.
    for (const auto& req : *requests) {
        EXPECT_EQ(req.op_type, OpType::kLinear);
        EXPECT_EQ(req.selector, MakeExpectedSelector());
    }
}

TEST(WeightPrepackPlannerTest, BuildRequestsExcludesNormsAndEmbeddings) {
    auto storage = std::make_shared<TestStorage>(256);
    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(1), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    for (const auto& req : *requests) {
        EXPECT_NE(req.raw_weight.data, index.embed_tokens.data);
        EXPECT_NE(req.raw_weight.data, index.final_norm.data);
        EXPECT_NE(req.raw_weight.data, index.layers[0].norm.input_rmsnorm.data);
        EXPECT_NE(req.raw_weight.data, index.layers[0].norm.post_attn_rmsnorm.data);
    }
}

TEST(WeightPrepackPlannerTest, BuildRequestsIncludesLmHeadWhenPresent) {
    auto storage = std::make_shared<TestStorage>(256);
    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.lm_head = MakeWeightView(storage, 16, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 24));

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            MakeLlamaConfig(1), index, backend, registry);

    ASSERT_TRUE(requests.ok());
    // 1 layer × 7 + lm_head = 8.
    EXPECT_EQ(requests->size(), 8);

    bool found_lm_head = false;
    for (const auto& req : *requests) {
        if (req.raw_weight.data == index.lm_head->data) {
            found_lm_head = true;
            break;
        }
    }
    EXPECT_TRUE(found_lm_head);
}

TEST(WeightPrepackPlannerTest, PrepackAndStoreMakesWeightsFindable) {
    auto storage = std::make_shared<TestStorage>(256);
    // Fill with zeros so Pack can safely memcpy.
    for (auto& b : storage->data) b = std::byte{0};

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(index));
    ASSERT_TRUE(model.ok());

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            (*model)->GetConfig(), (*model)->GetRawWeightIndex(), backend, registry);
    ASSERT_TRUE(requests.ok());

    Status status = WeightPrepackPlanner::PrepackAndStore(**model, *requests);
    ASSERT_TRUE(status.ok());

    const KernelSelector expected_selector = MakeExpectedSelector();
    const PackedWeights* found = (*model)->FindPackedWeights(
            OpType::kLinear, expected_selector);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->op_type(), OpType::kLinear);
    EXPECT_EQ(found->selector(), expected_selector);
    EXPECT_TRUE(found->storage().is_initialized());
}

TEST(WeightPrepackPlannerTest, PrepackAndStoreSkipsDuplicateSelectorWithoutError) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b : storage->data) b = std::byte{0};

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    // Two layers — all linear weights share the same (op_type, selector).
    index.layers.push_back(MakeTestLayer(storage, 16));
    index.layers.push_back(MakeTestLayer(storage, 100));

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(2), std::move(index));
    ASSERT_TRUE(model.ok());

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            (*model)->GetConfig(), (*model)->GetRawWeightIndex(), backend, registry);
    ASSERT_TRUE(requests.ok());
    EXPECT_EQ(requests->size(), 14);

    // PrepackAndStore should succeed — duplicates are skipped, not rejected.
    Status status = WeightPrepackPlanner::PrepackAndStore(**model, *requests);
    ASSERT_TRUE(status.ok());

    // Only the first unique (op_type, selector) pair is stored.
    const KernelSelector expected_selector = MakeExpectedSelector();
    const PackedWeights* found = (*model)->FindPackedWeights(
            OpType::kLinear, expected_selector);
    ASSERT_NE(found, nullptr);
}

TEST(WeightPrepackPlannerTest, RawViewsStillAccessibleAfterPrepack) {
    auto storage = std::make_shared<TestStorage>(256);
    for (auto& b : storage->data) b = std::byte{0};

    ModelWeightIndex index;
    index.embed_tokens = MakeWeightView(storage, 0, 8, DataType::Float32(), {2, 1});
    index.final_norm = MakeWeightView(storage, 8, 8, DataType::Float32(), {2, 1});
    index.layers.push_back(MakeTestLayer(storage, 16));

    auto model = ModelInstanceBuilder::Create(MakeLlamaConfig(1), std::move(index));
    ASSERT_TRUE(model.ok());

    CpuBackend backend;
    KernelRegistry registry;
    auto requests = WeightPrepackPlanner::BuildRequests(
            (*model)->GetConfig(), (*model)->GetRawWeightIndex(), backend, registry);
    ASSERT_TRUE(requests.ok());

    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(**model, *requests).ok());

    const auto& weight_index = (*model)->GetRawWeightIndex();
    EXPECT_TRUE(weight_index.embed_tokens.IsValid());
    EXPECT_TRUE(weight_index.final_norm.IsValid());
    EXPECT_TRUE(weight_index.layers[0].attn.q_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].ffn.down_proj.IsValid());
    EXPECT_TRUE(weight_index.layers[0].norm.input_rmsnorm.IsValid());
}

}  // namespace
}  // namespace aethermind
