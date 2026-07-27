#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <variant>
#include <vector>

namespace {
using namespace aethermind;

TEST(EmbeddingInference, ValidatesInputContract) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {2}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3, 2}).shape},
    };

    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingInference, InfersOutputShapeFromTokenIdsAndWeight) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {5}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {32000, 4096}).shape},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingInference, RejectsRankZeroTokenIds) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3, 2}).shape},
    };

    const Status status = InferOperator(OpType::kEmbedding, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingInference, RejectsRankZeroWeight) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {2}).shape},
            {.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };

    const Status status = InferOperator(OpType::kEmbedding, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingInference, AcceptsUint32TokenIds) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::UInt(32), .shape = MakeSpec(DataType::UInt(32), {2}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {3, 2}).shape},
    };

    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingInference, InfersOutputShapeWithUint32Tokens) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::UInt(32), .shape = MakeSpec(DataType::UInt(32), {5}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {32000, 4096}).shape},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingInference, PreservesSymbolicTokenAndHiddenDims) {
    constexpr EmbeddingParams params;
    const ShapeSymbol token_count = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{token_count})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden_size})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Symbolic weight dims (vocab, hidden) emit DimPositiveConstraint for runtime validation.
    ASSERT_EQ(inference->runtime_checks.size(), 2U);
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0], token_count);
    EXPECT_EQ(inference->outputs[0].shape[1], hidden_size);
}

TEST(EmbeddingInference, InfersOutputShapeForRank2Tokens) {
    // Rank-2 token_ids [batch, seq] must produce [batch, seq, hidden],
    // preserving all input axes (PyTorch nn.Embedding semantics).
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {2, 5}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {32000, 4096}).shape},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 4096);
}

TEST(EmbeddingInference, InfersOutputShapeForRank2SymbolicTokens) {
    // Symbolic rank-2 token_ids must preserve both symbolic axes in output.
    constexpr EmbeddingParams params;
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol seq = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, seq})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Symbolic weight dims (vocab, hidden) emit DimPositiveConstraint for runtime validation.
    ASSERT_EQ(inference->runtime_checks.size(), 2U);
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0], batch);
    EXPECT_EQ(inference->outputs[0].shape[1], seq);
    EXPECT_EQ(inference->outputs[0].shape[2], hidden);
}

TEST(EmbeddingInference, AcceptsZeroTokenCount) {
    // Zero token count yields empty output [0, hidden]; valid NumPy-style embedding.
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {0}).shape},
            {.dtype = DataType::Float32(), .shape = MakeSpec(DataType::Float32(), {32000, 4096}).shape},
    };
    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingInference, EmitsPositiveConstraintForSymbolicWeightDims) {
    // Symbolic weight dims must emit DimPositiveConstraint for runtime validation.
    constexpr EmbeddingParams params;
    const ShapeSymbol vocab = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = MakeSpec(DataType::Int(64), {3}).shape},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{vocab, hidden})},
    };
    const auto result = InferOperator(OpType::kEmbedding, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->runtime_checks.size(), 2U);

    // Verify DimPositiveConstraint locators without relying on ordering.
    bool found_vocab = false;
    bool found_hidden = false;
    for (const auto& check: result->runtime_checks) {
        const auto* pos = std::get_if<DimPositiveConstraint>(&check.condition);
        if (pos == nullptr) continue;
        if (pos->dim.tensor_port.tensor_idx == 1U && pos->dim.dim_index == 0U) {
            found_vocab = true;
        } else if (pos->dim.tensor_port.tensor_idx == 1U && pos->dim.dim_index == 1U) {
            found_hidden = true;
        }
    }
    EXPECT_TRUE(found_vocab) << "missing DimPositiveConstraint for weight vocab_size (input[1] dim[0])";
    EXPECT_TRUE(found_hidden) << "missing DimPositiveConstraint for weight hidden_size (input[1] dim[1])";
}

// --- Semantics ---

TEST(EmbeddingInference, AcceptsInt32TokenIds) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Float32(), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    EXPECT_TRUE(InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs).ok());
}

TEST(EmbeddingInference, InfersFloat32OutputForInt32Tokens) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Float32(), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    auto result = InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(EmbeddingInference, RejectsWrongWeightDtype) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Int(64), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    EXPECT_FALSE(InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs).ok());
}

}// namespace
