#include "aethermind/operators/attention_op.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// Phase-1 Attention logical signature:
//   q        : rank 2, [seq_len, hidden]  where hidden = num_attention_heads * head_dim
//   k/v cache: rank 3, [num_key_value_heads, cache_len, head_dim]
//   output   : rank 2, same spec as q

TEST(AttentionInference, SupportedDTypeSetIsExact) {
    ASSERT_EQ(kAttentionSupportedDTypes.size(), std::size_t{3});
    EXPECT_EQ(kAttentionSupportedDTypes[0], DataType::Float32());
    EXPECT_EQ(kAttentionSupportedDTypes[1], DataType::Float(16));
    EXPECT_EQ(kAttentionSupportedDTypes[2], DataType::BFloat(16));

    EXPECT_TRUE(IsAttentionSupportedDType(DataType::Float32()));
    EXPECT_TRUE(IsAttentionSupportedDType(DataType::Float(16)));
    EXPECT_TRUE(IsAttentionSupportedDType(DataType::BFloat(16)));

    EXPECT_FALSE(IsAttentionSupportedDType(DataType::Double()));
    EXPECT_FALSE(IsAttentionSupportedDType(DataType::Int(32)));
    EXPECT_FALSE(IsAttentionSupportedDType(DataType::Float8E4M3FN()));
    EXPECT_FALSE(IsAttentionSupportedDType(DataType::Float8E5M2()));

    EXPECT_EQ(MakeAttentionUnsupportedDTypeMessage("Attention q"),
              "Attention q only supports float32, float16, and bfloat16 dtypes");
}

TEST(AttentionInference, AcceptsValidParams) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(
                        OpType::kAttention,
                        AttentionParams{32, 8, 64}, inputs)
                        .ok());
}

TEST(AttentionInference, PreservesQSpecAsOutput) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention, p, attn_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), std::size_t{1});
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape, q.shape);
}

TEST(AttentionInference, AcceptsFloat16) {
    auto q = MakeSpec(DataType::Float(16), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float(16), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float(16), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention,
                              AttentionParams{32, 8, 64}, inputs)
                        .ok());
}

TEST(AttentionInference, AcceptsBFloat16) {
    auto q = MakeSpec(DataType::BFloat(16), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::BFloat(16), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::BFloat(16), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention,
                              AttentionParams{32, 8, 64}, inputs)
                        .ok());
}

// --- Params validation ---

TEST(AttentionInference, RejectsWrongParamsType) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention, std::monostate{}, inputs).ok());
}

TEST(AttentionInference, RejectsZeroNumAttentionHeads) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{0, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsZeroNumKeyValueHeads) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 0, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsZeroHeadDim) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 0}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsNonDivisibleHeads) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{7, 4, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsHeadOverflow) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{INT64_MAX, 8, 2}, inputs)
                         .ok());
}

// --- Arity validation ---

TEST(AttentionInference, RejectsWrongInputCount) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- Dtype validation ---

TEST(AttentionInference, RejectsInt32Q) {
    auto q = MakeSpec(DataType::Int(32), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Int(32), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Int(32), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsFloat8Q) {
    auto q = MakeSpec(DataType::Float8E4M3FN(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float8E4M3FN(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float8E4M3FN(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsMismatchedKVAndQDtype) {
    auto q = MakeSpec(DataType::Float(16), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsMismatchedCacheDtype) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Double(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- Rank validation ---

TEST(AttentionInference, RejectsRank3Q) {
    auto q = MakeSpec(DataType::Float32(), {1, 1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsRank2Cache) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- Shape equation validation ---

TEST(AttentionInference, RejectsQHiddenMismatch) {
    // q last dim = 100 but params say hidden = 32*64 = 2048.
    auto q = MakeSpec(DataType::Float32(), {1, 100});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsKVHeadsMismatch) {
    // cache kv_heads = 4 but params say num_key_value_heads = 8.
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {4, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {4, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsHeadDimMismatch) {
    // cache head_dim = 32 but params say head_dim = 64.
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 32});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 32});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsKAndVCacheShapeMismatch) {
    // v_cache has different cache_len than k_cache.
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 2048, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- Static zero rejection ---

TEST(AttentionInference, RejectsZeroSeqLen) {
    auto q = MakeSpec(DataType::Float32(), {0, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

TEST(AttentionInference, RejectsZeroCacheLen) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 0, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 0, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- Symbolic dim acceptance ---

TEST(AttentionInference, AcceptsSymbolicSeqLen) {
    // q seq_len is symbolic; must pass and emit DimPositiveConstraint.
    const ShapeSymbol sym_seq = ShapeSymbol::Create();
    const SymbolicShape q_shape{sym_seq, ShapeSymbol::CreateFromValue(32 * 64)};
    const TensorSpec q{DataType::Float32(), q_shape};
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention,
                                AttentionParams{32, 8, 64}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Exactly one DimPositiveConstraint for input port 0, dim 0.
    ASSERT_EQ(result->runtime_checks.size(), std::size_t{1});
    const auto& constraint = result->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimPositiveConstraint>(constraint.condition));
    const auto& dim_pos = std::get<DimPositiveConstraint>(constraint.condition);
    EXPECT_EQ(dim_pos.dim.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(dim_pos.dim.tensor_port.tensor_idx, std::size_t{0});
    EXPECT_EQ(dim_pos.dim.dim_index, std::size_t{0});
}

TEST(AttentionInference, AcceptsSymbolicCacheLen) {
    // cache_len is symbolic; k_cache and v_cache share the same symbol.
    const ShapeSymbol sym_cache = ShapeSymbol::Create();
    const SymbolicShape cache_shape{
            ShapeSymbol::CreateFromValue(8), sym_cache,
            ShapeSymbol::CreateFromValue(64)};
    const TensorSpec kCache{DataType::Float32(), cache_shape};
    const TensorSpec vCache{DataType::Float32(), cache_shape};
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention,
                              AttentionParams{32, 8, 64}, inputs)
                        .ok());
}

TEST(AttentionInference, AcceptsSymbolicSeqLenAndCacheLen) {
    // Both seq_len and cache_len are symbolic; no static zero rejection.
    const ShapeSymbol sym_seq = ShapeSymbol::Create();
    const ShapeSymbol sym_cache = ShapeSymbol::Create();
    const SymbolicShape q_shape{sym_seq, ShapeSymbol::CreateFromValue(32 * 64)};
    const SymbolicShape cache_shape{
            ShapeSymbol::CreateFromValue(8), sym_cache,
            ShapeSymbol::CreateFromValue(64)};
    const TensorSpec q{DataType::Float32(), q_shape};
    const TensorSpec kCache{DataType::Float32(), cache_shape};
    const TensorSpec vCache{DataType::Float32(), cache_shape};
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention,
                                AttentionParams{32, 8, 64}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    // Exactly one DimPositiveConstraint for symbolic seq_len.
    EXPECT_EQ(result->runtime_checks.size(), std::size_t{1});
}

TEST(AttentionInference, RejectsSymbolicCacheLenMismatch) {
    // k_cache and v_cache use different symbolic cache_len dims.
    const ShapeSymbol sym_cache_k = ShapeSymbol::Create();
    const ShapeSymbol sym_cache_v = ShapeSymbol::Create();
    const SymbolicShape k_shape{
            ShapeSymbol::CreateFromValue(8), sym_cache_k,
            ShapeSymbol::CreateFromValue(64)};
    const SymbolicShape v_shape{
            ShapeSymbol::CreateFromValue(8), sym_cache_v,
            ShapeSymbol::CreateFromValue(64)};
    const TensorSpec q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    const TensorSpec kCache{DataType::Float32(), k_shape};
    const TensorSpec vCache{DataType::Float32(), v_shape};
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    EXPECT_FALSE(InferOperator(OpType::kAttention,
                               AttentionParams{32, 8, 64}, inputs)
                         .ok());
}

// --- No constraints when all dims are static ---

TEST(AttentionInference, EmitsNoConstraintsWhenAllStatic) {
    auto q = MakeSpec(DataType::Float32(), {1, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention,
                                AttentionParams{32, 8, 64}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_TRUE(result->runtime_checks.empty());
}

}// namespace
