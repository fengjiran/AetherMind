#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/inference/executable_model.h"
#include "aethermind/model/weight/weight_packing_request.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "test_utils/test_resource_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

using namespace aethermind;

struct WeightStorage final : RawStorage {
    alignas(64) std::array<float, 4> values{1.0F, 2.0F, 3.0F, 4.0F};

    std::optional<RawStorageFootprint> footprint() const noexcept override {
        return RawStorageFootprint{RawStorageKind::kExternal, sizeof(values)};
    }
};

struct InvalidElementType {
    const char* name;
    DataType dtype;
    size_t bytes;
};

std::array<InvalidElementType, 7> InvalidElementTypes() {
    return {{{"ScalableFloat32", DataType(DLDataTypeCode::kFloat, 32, 2, true), 16},
             {"ScalableFloat16", DataType(DLDataTypeCode::kFloat, 16, 2, true), 8},
             {"ScalableBFloat16", DataType(DLDataTypeCode::kBFloat, 16, 2, true), 8},
             {"ReservedLaneEncoding", DataType::Float(32, 65535), 16},
             {"ZeroLaneHandle", DataType::Handle(64, 0), 0},
             {"ZeroByteHandle", DataType::Handle(0, 1), 0},
             {"Undefined", DataType{}, 0}}};
}

RawWeightView MakeRawWeight(const std::shared_ptr<WeightStorage>& storage,
                            DataType dtype = DataType::Float32(), size_t bytes = 16) {
    return {.data = reinterpret_cast<const std::byte*>(storage->values.data()),
            .bytes = bytes,
            .dtype = dtype,
            .shape = {2, 2},
            .storage = storage};
}

Runtime MakeRecordingRuntime(const std::shared_ptr<test::ResourceAllocationStats>& stats) {
    RuntimeBuilder builder;
    builder.RegisterCustomAllocatorProvider(DeviceType::kCPU,
                                            std::make_unique<test::ResourceRecordingProvider>(stats));
    return builder.Build();
}

WeightPackingRequest MakeEmbeddingRequest(RawWeightView raw) {
    return {.op_type = OpType::kEmbedding,
            .source_id = 1,
            .value_index = 0,
            .binding = MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding),
            .components = {std::move(raw)},
            .selector = {.device_type = DeviceType::kCPU,
                         .act_dtype = DataType::Float32(),
                         .weight_dtype = DataType::Float32(),
                         .weight_format = WeightFormat::kPacked,
                         .phase = ExecPhase::kBoth},
            .recipe = CpuIdentityPackingRecipe()};
}

StatusOr<LoweredModelArtifact> MakeEmbeddingArtifact(RawWeightView raw, bool packed) {
    // The graph has valid FP32 semantics; the untrusted physical weight metadata
    // must be rejected by preparation before it reaches a checked TensorView.
    ModelGraph graph;
    const std::array<int64_t, 2> weight_shape{2, 2};
    const std::array<int64_t, 1> token_shape{1};
    const auto weight = graph.AddWeight(
            {.dtype = DataType::Float32(), .shape = SymbolicShape(weight_shape)},
            MakeTransformerWeightBinding(std::nullopt, TransformerWeightRole::kTokenEmbedding));
    const auto tokens = graph.AddInput({.dtype = DataType::Int(64), .shape = SymbolicShape(token_shape)});
    AM_ASSIGN_OR_RETURN(auto embedded,
                        graph.AddNode(OpType::kEmbedding, std::nullopt, {tokens, weight},
                                      {NodeOutputDesc{.payload = ActivationValue{}}}, EmbeddingParams{}));
    graph.MarkOutput(embedded.outputs.front());
    GraphLoweringConfig lowering;
    lowering.enable_packed_weights = packed;
    AM_ASSIGN_OR_RETURN(auto lowered, LowerModelGraph(graph, lowering));
    HfModelConfig config;
    config.vocab_size = 2;
    config.max_position_embeddings = 4;
    ResolvedModelWeights weights;
    weights.embed_tokens = std::move(raw);
    return LoweredModelArtifact{.loaded_model = std::make_unique<LoadedModel>(config, std::move(weights)),
                                .graph = std::move(lowered)};
}

TEST(RawWeightConsumers, PrepackRejectsInvalidElementDTypesBeforeAllocation) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeRecordingRuntime(stats);
    auto backend = runtime.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend.ok()) << backend.status().ToString();
    auto& allocator = runtime.GetAllocator(Device::CPU());
    auto storage = std::make_shared<WeightStorage>();

    for (const auto& invalid: InvalidElementTypes()) {
        SCOPED_TRACE(invalid.name);
        const auto request = MakeEmbeddingRequest(MakeRawWeight(storage, invalid.dtype, invalid.bytes));
        for (const auto cap: {std::optional<size_t>{}, std::optional<size_t>{64}}) {
            SCOPED_TRACE(cap.has_value() ? "Bounded" : "Unlimited");
            stats->requests.clear();
            const auto result = PrepackWeightRequests(**backend, {request}, allocator, cap);
            EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument) << result.status().ToString();
            EXPECT_TRUE(stats->requests.empty());
        }
    }

    const auto retry = PrepackWeightRequests(**backend, {MakeEmbeddingRequest(MakeRawWeight(storage))}, allocator);
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_EQ(retry->size(), 1U);
    EXPECT_FALSE(stats->requests.empty());
    EXPECT_EQ(storage->values, (std::array<float, 4>{1.0F, 2.0F, 3.0F, 4.0F}));
}

TEST(RawWeightConsumers, PlainPreparationRejectsInvalidElementDTypesBeforeBinding) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeRecordingRuntime(stats);
    auto storage = std::make_shared<WeightStorage>();

    for (const auto& invalid: InvalidElementTypes()) {
        SCOPED_TRACE(invalid.name);
        auto artifact = MakeEmbeddingArtifact(MakeRawWeight(storage, invalid.dtype, invalid.bytes), false);
        ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
        stats->requests.clear();
        const auto result = PrepareExecutableModel(runtime, std::move(*artifact));
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument) << result.status().ToString();
        EXPECT_TRUE(stats->requests.empty());
    }

    auto artifact = MakeEmbeddingArtifact(MakeRawWeight(storage), false);
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
    const auto retry = PrepareExecutableModel(runtime, std::move(*artifact));
    EXPECT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_TRUE(stats->requests.empty());
}

TEST(RawWeightConsumers, PackedPreparationRejectsInvalidElementDTypesBeforeAllocation) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    Runtime runtime = MakeRecordingRuntime(stats);
    auto storage = std::make_shared<WeightStorage>();

    for (const auto& invalid: InvalidElementTypes()) {
        SCOPED_TRACE(invalid.name);
        auto artifact = MakeEmbeddingArtifact(MakeRawWeight(storage, invalid.dtype, invalid.bytes), true);
        ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
        stats->requests.clear();
        const auto result = PrepareExecutableModel(runtime, std::move(*artifact));
        EXPECT_EQ(result.status().code(), StatusCode::kInvalidArgument) << result.status().ToString();
        EXPECT_TRUE(stats->requests.empty());
    }

    auto artifact = MakeEmbeddingArtifact(MakeRawWeight(storage), true);
    ASSERT_TRUE(artifact.ok()) << artifact.status().ToString();
    const auto retry = PrepareExecutableModel(runtime, std::move(*artifact));
    EXPECT_TRUE(retry.ok()) << retry.status().ToString();
    EXPECT_FALSE(stats->requests.empty());
}

} // namespace
