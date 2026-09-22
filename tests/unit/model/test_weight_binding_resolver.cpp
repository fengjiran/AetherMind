#include "aethermind/model/weight_binding_resolver.h"

#include "aethermind/graph/graph_types.h"
#include "aethermind/model/resolved_model_weights.h"

#include <cstddef>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <vector>

namespace {

using namespace aethermind;

struct TestStorage : RawStorage {
    explicit TestStorage(size_t nbytes) : data(nbytes) {}
    std::vector<std::byte> data;
};

constexpr size_t kViewBytes = 8;
constexpr size_t kLayerStride = 72;

RawWeightView MakeWeightView(const std::shared_ptr<TestStorage>& storage, size_t offset) {
    return RawWeightView{
            .data = storage->data.data() + offset,
            .bytes = kViewBytes,
            .dtype = DataType::Float32(),
            .shape = {2, 1},
            .storage = storage,
    };
}

DecoderLayerRawWeights MakeLayer(const std::shared_ptr<TestStorage>& storage, size_t base_offset) {
    DecoderLayerRawWeights layer;
    layer.norm.input_rmsnorm = MakeWeightView(storage, base_offset + 0);
    layer.norm.post_attn_rmsnorm = MakeWeightView(storage, base_offset + 8);
    layer.attn.q_proj = MakeWeightView(storage, base_offset + 16);
    layer.attn.k_proj = MakeWeightView(storage, base_offset + 24);
    layer.attn.v_proj = MakeWeightView(storage, base_offset + 32);
    layer.attn.o_proj = MakeWeightView(storage, base_offset + 40);
    layer.mlp.gate_proj = MakeWeightView(storage, base_offset + 48);
    layer.mlp.up_proj = MakeWeightView(storage, base_offset + 56);
    layer.mlp.down_proj = MakeWeightView(storage, base_offset + 64);
    return layer;
}

/// Two-layer weights. `with_lm_head` selects an untied checkpoint; omitting it
/// produces the tied shape whose lm_head must reuse embed_tokens.
ResolvedModelWeights MakeResolved(const std::shared_ptr<TestStorage>& storage,
                                  bool with_lm_head) {
    ResolvedModelWeights resolved;
    resolved.embed_tokens = MakeWeightView(storage, 0);
    resolved.final_norm = MakeWeightView(storage, kViewBytes);
    if (with_lm_head) {
        resolved.lm_head = MakeWeightView(storage, 2 * kViewBytes);
    }
    resolved.layers.push_back(MakeLayer(storage, 3 * kViewBytes));
    resolved.layers.push_back(MakeLayer(storage, 3 * kViewBytes + kLayerStride));
    return resolved;
}

WeightBinding RoleBinding(std::optional<uint32_t> layer, TransformerWeightRole role) {
    return MakeTransformerWeightBinding(layer, role);
}

TEST(WeightBindingResolver, ModelLevelRolesResolveToTheirOwnWeight) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(
            ResolveWeightBinding(
                    RoleBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding), resolved),
            &resolved.embed_tokens);
    EXPECT_EQ(
            ResolveWeightBinding(
                    RoleBinding(std::nullopt, TransformerWeightRole::kFinalNorm), resolved),
            &resolved.final_norm);
}

TEST(WeightBindingResolver, UntiedLmHeadResolvesToItsOwnWeight) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    ASSERT_TRUE(resolved.lm_head.has_value());

    const RawWeightView* lm_head =
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                 resolved);
    ASSERT_NE(lm_head, nullptr);
    EXPECT_EQ(lm_head, &*resolved.lm_head);
    EXPECT_NE(lm_head->data, resolved.embed_tokens.data);
}

TEST(WeightBindingResolver, TiedLmHeadReusesEmbedTokensBacking) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/false);
    ASSERT_FALSE(resolved.lm_head.has_value());

    const RawWeightView* lm_head =
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kLmHead),
                                 resolved);
    ASSERT_NE(lm_head, nullptr);
    EXPECT_EQ(lm_head, &resolved.embed_tokens);
    EXPECT_EQ(lm_head->data, resolved.embed_tokens.data);
    EXPECT_EQ(lm_head->storage, resolved.embed_tokens.storage);
}

TEST(WeightBindingResolver, LayerScopedRolesResolveWithinTheIndexedLayer) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    const DecoderLayerRawWeights& layer = resolved.layers[1];

    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kInputNorm), resolved),
              &layer.norm.input_rmsnorm);
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kPostAttentionNorm),
                                 resolved),
            &layer.norm.post_attn_rmsnorm);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionQ), resolved),
              &layer.attn.q_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionK), resolved),
              &layer.attn.k_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionV), resolved),
              &layer.attn.v_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionO), resolved),
              &layer.attn.o_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpGate), resolved),
              &layer.mlp.gate_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpUp), resolved),
              &layer.mlp.up_proj);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kMlpDown), resolved),
              &layer.mlp.down_proj);
}

TEST(WeightBindingResolver, LayerIndexSelectsTheRequestedLayer) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    const RawWeightView* layer0 =
            ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kAttentionQ), resolved);
    const RawWeightView* layer1 =
            ResolveWeightBinding(RoleBinding(1, TransformerWeightRole::kAttentionQ), resolved);
    ASSERT_NE(layer0, nullptr);
    ASSERT_NE(layer1, nullptr);
    EXPECT_EQ(layer0, &resolved.layers[0].attn.q_proj);
    EXPECT_NE(layer0, layer1);
}

TEST(WeightBindingResolver, OutOfRangeLayerIndexReturnsNull) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);
    ASSERT_EQ(resolved.layers.size(), 2U);

    for (const uint32_t layer: {2U, 3U, 1000U}) {
        EXPECT_EQ(ResolveWeightBinding(RoleBinding(layer, TransformerWeightRole::kAttentionQ),
                                       resolved),
                  nullptr)
                << "layer " << layer;
        EXPECT_EQ(ResolveWeightBinding(RoleBinding(layer, TransformerWeightRole::kInputNorm),
                                       resolved),
                  nullptr)
                << "layer " << layer;
    }
}

TEST(WeightBindingResolver, MissingLayerIndexReturnsNullForLayerScopedRoles) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    // ModelGraph::Validate rejects a layer-scoped role without a layer index,
    // so the resolver must report null instead of silently binding layer 0.
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kAttentionQ),
                                 resolved),
            nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kMlpDown),
                                   resolved),
              nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kInputNorm),
                                   resolved),
              nullptr);
    EXPECT_EQ(
            ResolveWeightBinding(RoleBinding(std::nullopt, TransformerWeightRole::kPostAttentionNorm),
                                 resolved),
            nullptr);
}

TEST(WeightBindingResolver, ModelLevelRolesIgnoreLayerIndex) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    // Layer-scope validation belongs to ModelGraph::Validate; resolution of a
    // model-level role never depends on the layer index.
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(7, TransformerWeightRole::kTokenEmbedding), resolved),
              &resolved.embed_tokens);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(7, TransformerWeightRole::kFinalNorm), resolved),
              &resolved.final_norm);
}

TEST(WeightBindingResolver, MoeRouterHasNoDenseStorage) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kMoERouter), resolved),
              nullptr);
}

TEST(WeightBindingResolver, CompositeBindingsHaveNoSingleResolution) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(MakeQkvWeightBinding(0), resolved), nullptr);
    EXPECT_EQ(ResolveWeightBinding(MakeGateUpWeightBinding(0), resolved), nullptr);
}

TEST(WeightBindingResolver, RolelessDirectBindingReturnsNull) {
    auto storage = std::make_shared<TestStorage>(256);
    const ResolvedModelWeights resolved = MakeResolved(storage, /*with_lm_head=*/true);

    EXPECT_EQ(ResolveWeightBinding(MakeDirectWeightBinding(ParameterSlot::kKernel), resolved),
              nullptr);
}

TEST(WeightBindingResolver, EmptyLayerListRejectsLayerScopedRoles) {
    const ResolvedModelWeights resolved;

    EXPECT_EQ(resolved.layers.size(), 0U);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kAttentionQ), resolved),
              nullptr);
    EXPECT_EQ(ResolveWeightBinding(RoleBinding(0, TransformerWeightRole::kInputNorm), resolved),
              nullptr);
}

} // namespace
