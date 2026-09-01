#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/ops/rope_op.h"
#include "aethermind/shape_inference/shape_constraint.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <limits>

namespace {
using namespace aethermind;

constexpr int64_t kHeadDim = 64;
constexpr int64_t kNumAttentionHeads = 32;
constexpr int64_t kNumKeyValueHeads = 8;
constexpr int64_t kMaxPositionEmbeddings = 2048;

RoPEParams MakeStandardParams() {
    return RoPEParams{
            .head_dim = kHeadDim,
            .num_attention_heads = kNumAttentionHeads,
            .num_key_value_heads = kNumKeyValueHeads,
            .max_position_embeddings = kMaxPositionEmbeddings,
            .theta = 10000.0,
            .scaling_factor = std::nullopt,
            .scaling_type = RoPEScalingType::kNone,
    };
}

std::vector<TensorSpec> MakeInputs(DataType dtype) {
    return {
            MakeSpec(dtype, {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(dtype, {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
}

TEST(RoPEInference, SupportedDTypeSetIsExact) {
    ASSERT_EQ(kRoPESupportedDTypes.size(), std::size_t{3});
    EXPECT_EQ(kRoPESupportedDTypes[0], DataType::Float32());
    EXPECT_EQ(kRoPESupportedDTypes[1], DataType::Float(16));
    EXPECT_EQ(kRoPESupportedDTypes[2], DataType::BFloat(16));
    EXPECT_TRUE(IsRoPESupportedDType(DataType::Float32()));
    EXPECT_TRUE(IsRoPESupportedDType(DataType::Float(16)));
    EXPECT_TRUE(IsRoPESupportedDType(DataType::BFloat(16)));
    EXPECT_FALSE(IsRoPESupportedDType(DataType::Double()));
    EXPECT_FALSE(IsRoPESupportedDType(DataType::Int(32)));
    EXPECT_FALSE(IsRoPESupportedDType(DataType::Float8E4M3FN()));
    EXPECT_FALSE(IsRoPESupportedDType(DataType::Float8E5M2()));
    EXPECT_EQ(MakeRoPEUnsupportedDTypeMessage("RoPE q"),
              "RoPE q only supports float32, float16, and bfloat16 dtypes");
}

TEST(RoPEInference, RejectsWrongParamsType) {
    std::vector<TensorSpec> inputs = MakeInputs(DataType::Float32());
    EXPECT_FALSE(InferOperator(OpType::kRoPE, std::monostate{}, inputs).ok());
}
// --- Params invariants ---

TEST(RoPEInference, RejectsZeroHeadDim) {
    auto p = MakeStandardParams();
    p.head_dim = 0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsZeroNumAttentionHeads) {
    auto p = MakeStandardParams();
    p.num_attention_heads = 0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsZeroNumKeyValueHeads) {
    auto p = MakeStandardParams();
    p.num_key_value_heads = 0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsZeroMaxPositionEmbeddings) {
    auto p = MakeStandardParams();
    p.max_position_embeddings = 0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsOddHeadDim) {
    auto p = MakeStandardParams();
    p.head_dim = 63;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsZeroTheta) {
    auto p = MakeStandardParams();
    p.theta = 0.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsNanTheta) {
    auto p = MakeStandardParams();
    p.theta = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsNegativeTheta) {
    auto p = MakeStandardParams();
    p.theta = -1.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsAttentionHeadsProductOverflow) {
    auto p = MakeStandardParams();
    p.num_attention_heads = INT64_MAX;
    p.head_dim = 2;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsKeyValueHeadsProductOverflow) {
    auto p = MakeStandardParams();
    p.num_key_value_heads = INT64_MAX;
    p.head_dim = 2;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}
// --- Arity validation ---

TEST(RoPEInference, RejectsWrongInputCount) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

// --- Dtype validation ---

TEST(RoPEInference, AcceptsFloat32) {
    auto p = MakeStandardParams();
    auto result = InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32()));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), std::size_t{2});
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(RoPEInference, AcceptsFloat16) {
    auto p = MakeStandardParams();
    auto result = InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float(16)));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(RoPEInference, AcceptsBFloat16) {
    auto p = MakeStandardParams();
    auto result = InferOperator(OpType::kRoPE, p, MakeInputs(DataType::BFloat(16)));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(RoPEInference, RejectsInt32Q) {
    auto p = MakeStandardParams();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Int(32))).ok());
}

TEST(RoPEInference, RejectsFloat8Q) {
    auto p = MakeStandardParams();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float8E4M3FN())).ok());
}

TEST(RoPEInference, RejectsMismatchedQAndKDtype) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float(16), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsWrongPositionIdsDtype) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}
// --- Scaling tuple validation ---

TEST(RoPEInference, AcceptsStandardScaling) {
    auto p = MakeStandardParams();
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsMisplacedFactorOnStandardScaling) {
    auto p = MakeStandardParams();
    p.scaling_factor = 2.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, AcceptsLinearScaling) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = 2.0;
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, AcceptsLinearScalingFactorOne) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = 1.0;
    auto result = InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32()));
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(p.scaling_type, RoPEScalingType::kLinear);
    EXPECT_TRUE(p.scaling_factor.has_value());
    EXPECT_EQ(*p.scaling_factor, 1.0);
}

TEST(RoPEInference, AcceptsLinearScalingFactorBelowOne) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = 0.5;
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}
TEST(RoPEInference, RejectsLinearScalingMissingFactor) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = std::nullopt;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsLinearScalingNonFiniteFactor) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = std::numeric_limits<double>::infinity();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
    p.scaling_factor = std::numeric_limits<double>::quiet_NaN();
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

TEST(RoPEInference, RejectsLinearScalingNonPositiveFactor) {
    auto p = MakeStandardParams();
    p.scaling_type = RoPEScalingType::kLinear;
    p.scaling_factor = 0.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
    p.scaling_factor = -1.0;
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, MakeInputs(DataType::Float32())).ok());
}

// HF-only RoPE scaling variants (kDynamicNtk, kYarn, kLlama3, kLongRope,
// kSu, kUnknown) are not representable on the semantic RoPEScalingType
// surface. Rejection of these variants is exercised at the model frontend
// boundary in tests/unit/model/test_model_graph_builder.cpp
// (ModelGraphBuilder.RejectsUnsupportedRoPE*).

// --- Output preservation ---

TEST(RoPEInference, PreservesInputSpecsAsOutputs) {
    auto p = MakeStandardParams();
    auto inputs = MakeInputs(DataType::Float32());
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->outputs.size(), std::size_t{2});
    EXPECT_EQ(result->outputs[0].dtype, inputs[0].dtype);
    EXPECT_EQ(result->outputs[0].shape, inputs[0].shape);
    EXPECT_EQ(result->outputs[1].dtype, inputs[1].dtype);
    EXPECT_EQ(result->outputs[1].shape, inputs[1].shape);
}

// --- Rank validation ---

TEST(RoPEInference, RejectsQRank3) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {1, 128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsKRank3) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {1, 128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsPositionIdsRank0) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsPositionIdsRank2) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), {128, 1}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsUnrankedQ) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32()),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsUnrankedK) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32()),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsUnrankedPositionIds) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64)),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

// --- Static width equations ---

TEST(RoPEInference, RejectsQStaticWidthMismatch) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim + 1}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsKStaticWidthMismatch) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim + 1}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, AcceptsSymbolicQWidth) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_hidden = ShapeSymbol::Create();
    const TensorSpec q{DataType::Float32(),
                       SymbolicShape{ShapeSymbol::CreateFromValue(128), sym_hidden}};
    std::vector<TensorSpec> inputs = {
            q,
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, AcceptsSymbolicKWidth) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_hidden = ShapeSymbol::Create();
    const TensorSpec k{DataType::Float32(),
                       SymbolicShape{ShapeSymbol::CreateFromValue(128), sym_hidden}};
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            k,
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

// --- Sequence validation ---

TEST(RoPEInference, RejectsZeroSeqLen) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {0, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {0, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{0}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsStaticQKSeqLenMismatch) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {256, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{128}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

TEST(RoPEInference, RejectsStaticQPositionSeqLenMismatch) {
    auto p = MakeStandardParams();
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {128, kNumAttentionHeads * kHeadDim}),
            MakeSpec(DataType::Float32(), {128, kNumKeyValueHeads * kHeadDim}),
            MakeSpec(DataType::Int(64), std::vector<int64_t>{256}),
    };
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, inputs).ok());
}

// --- Valid rank-2 contract and runtime-check emission ---

TEST(RoPEInference, AcceptsValidRank2Contract) {
    auto p = MakeStandardParams();
    auto inputs = MakeInputs(DataType::Float32());
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(RoPEInference, EmitsNoConstraintsWhenAllStatic) {
    auto p = MakeStandardParams();
    auto inputs = MakeInputs(DataType::Float32());
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(RoPEInference, AcceptsSharedSymbolicSequence) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_seq = ShapeSymbol::Create();
    const TensorSpec q{
            DataType::Float32(),
            SymbolicShape{sym_seq, ShapeSymbol::CreateFromValue(kNumAttentionHeads * kHeadDim)}};
    const TensorSpec k{
            DataType::Float32(),
            SymbolicShape{sym_seq, ShapeSymbol::CreateFromValue(kNumKeyValueHeads * kHeadDim)}};
    const TensorSpec pos{DataType::Int(64), SymbolicShape{sym_seq}};
    std::vector<TensorSpec> inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Shared symbolic seq_len emits exactly one positivity check; no equality
    // checks because q/k/position_ids share the same symbol.
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{1});
    const auto& positivity = result->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimPositiveConstraint>(positivity.condition));
    const auto& dim_pos = std::get<DimPositiveConstraint>(positivity.condition);
    EXPECT_EQ(dim_pos.dim.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(dim_pos.dim.tensor_port.tensor_idx, std::size_t{0});
    EXPECT_EQ(dim_pos.dim.dim_index, std::size_t{0});
}

TEST(RoPEInference, EmitsChecksForIndependentSymbolicSequences) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_q = ShapeSymbol::Create();
    const ShapeSymbol sym_k = ShapeSymbol::Create();
    const ShapeSymbol sym_pos = ShapeSymbol::Create();
    const TensorSpec q{
            DataType::Float32(),
            SymbolicShape{sym_q, ShapeSymbol::CreateFromValue(kNumAttentionHeads * kHeadDim)}};
    const TensorSpec k{
            DataType::Float32(),
            SymbolicShape{sym_k, ShapeSymbol::CreateFromValue(kNumKeyValueHeads * kHeadDim)}};
    const TensorSpec pos{DataType::Int(64), SymbolicShape{sym_pos}};
    std::vector<TensorSpec> inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Three distinct symbolic seq_len dims: one positivity check for q
    // followed by two equality checks (q vs k, then q vs position_ids).
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{3});

    const auto& check0 = result->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimPositiveConstraint>(check0.condition));
    {
        const auto& dim_pos = std::get<DimPositiveConstraint>(check0.condition);
        EXPECT_EQ(dim_pos.dim.tensor_port.direction, TensorPortType::kInput);
        EXPECT_EQ(dim_pos.dim.tensor_port.tensor_idx, std::size_t{0});
        EXPECT_EQ(dim_pos.dim.dim_index, std::size_t{0});
    }

    const auto& check1 = result->runtime_checks[1];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(check1.condition));
    {
        const auto& eq = std::get<DimEqualConstraint>(check1.condition);
        EXPECT_EQ(eq.lhs.tensor_port.direction, TensorPortType::kInput);
        EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, std::size_t{0});
        EXPECT_EQ(eq.lhs.dim_index, std::size_t{0});
        EXPECT_EQ(eq.rhs.tensor_port.direction, TensorPortType::kInput);
        EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, std::size_t{1});
        EXPECT_EQ(eq.rhs.dim_index, std::size_t{0});
    }

    const auto& check2 = result->runtime_checks[2];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(check2.condition));
    {
        const auto& eq = std::get<DimEqualConstraint>(check2.condition);
        EXPECT_EQ(eq.lhs.tensor_port.direction, TensorPortType::kInput);
        EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, std::size_t{0});
        EXPECT_EQ(eq.lhs.dim_index, std::size_t{0});
        EXPECT_EQ(eq.rhs.tensor_port.direction, TensorPortType::kInput);
        EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, std::size_t{2});
        EXPECT_EQ(eq.rhs.dim_index, std::size_t{0});
    }
}

TEST(RoPEInference, EmitsEqualityCheckForQKDistinctSymbols) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_q = ShapeSymbol::Create();
    const ShapeSymbol sym_k = ShapeSymbol::Create();
    const TensorSpec q{
            DataType::Float32(),
            SymbolicShape{sym_q, ShapeSymbol::CreateFromValue(kNumAttentionHeads * kHeadDim)}};
    const TensorSpec k{
            DataType::Float32(),
            SymbolicShape{sym_k, ShapeSymbol::CreateFromValue(kNumKeyValueHeads * kHeadDim)}};
    const TensorSpec pos{DataType::Int(64), SymbolicShape{sym_q}};
    std::vector<TensorSpec> inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // q and position_ids share sym_q; only q vs k equality check is emitted.
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{2});
    ASSERT_TRUE(std::holds_alternative<DimPositiveConstraint>(result->runtime_checks[0].condition));
    const auto& eq = std::get<DimEqualConstraint>(result->runtime_checks[1].condition);
    EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, std::size_t{0});
    EXPECT_EQ(eq.lhs.dim_index, std::size_t{0});
    EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, std::size_t{1});
    EXPECT_EQ(eq.rhs.dim_index, std::size_t{0});
}

TEST(RoPEInference, EmitsEqualityCheckForQPositionDistinctSymbols) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_q = ShapeSymbol::Create();
    const ShapeSymbol sym_pos = ShapeSymbol::Create();
    const TensorSpec q{
            DataType::Float32(),
            SymbolicShape{sym_q, ShapeSymbol::CreateFromValue(kNumAttentionHeads * kHeadDim)}};
    const TensorSpec k{
            DataType::Float32(),
            SymbolicShape{sym_q, ShapeSymbol::CreateFromValue(kNumKeyValueHeads * kHeadDim)}};
    const TensorSpec pos{DataType::Int(64), SymbolicShape{sym_pos}};
    std::vector<TensorSpec> inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // q and k share sym_q; only q vs position_ids equality check is emitted.
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{2});
    ASSERT_TRUE(std::holds_alternative<DimPositiveConstraint>(result->runtime_checks[0].condition));
    const auto& eq = std::get<DimEqualConstraint>(result->runtime_checks[1].condition);
    EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, std::size_t{0});
    EXPECT_EQ(eq.lhs.dim_index, std::size_t{0});
    EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, std::size_t{2});
    EXPECT_EQ(eq.rhs.dim_index, std::size_t{0});
}

TEST(RoPEInference, EmitsEqualityCheckForStaticSymbolicMixQK) {
    auto p = MakeStandardParams();
    const ShapeSymbol sym_k = ShapeSymbol::Create();
    const TensorSpec q{DataType::Float32(),
                       SymbolicShape{ShapeSymbol::CreateFromValue(128),
                                     ShapeSymbol::CreateFromValue(kNumAttentionHeads * kHeadDim)}};
    const TensorSpec k{
            DataType::Float32(),
            SymbolicShape{sym_k, ShapeSymbol::CreateFromValue(kNumKeyValueHeads * kHeadDim)}};
    const TensorSpec pos{DataType::Int(64), SymbolicShape{ShapeSymbol::CreateFromValue(128)}};
    std::vector<TensorSpec> inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // q seq_len is static positive: no positivity check. q vs k emits one
    // equality check; q vs position_ids are both static 128: no check.
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{1});
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(result->runtime_checks[0].condition));
    const auto& eq = std::get<DimEqualConstraint>(result->runtime_checks[0].condition);
    EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, std::size_t{0});
    EXPECT_EQ(eq.lhs.dim_index, std::size_t{0});
    EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, std::size_t{1});
    EXPECT_EQ(eq.rhs.dim_index, std::size_t{0});
}

} // namespace
