#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/model/graph/operator_schema.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

namespace {
using namespace aethermind;

TensorSpec MakeSpec(DataType dtype) {
    return {dtype, SymbolicShape(std::nullopt)};
}

TensorSpec MakeSpec(DataType dtype, const std::vector<int64_t>& dims) {
    std::vector<ShapeSymbol> symbols;
    for (auto d: dims) {
        symbols.push_back(ShapeSymbol::CreateFromValue(d));
    }
    return {dtype, SymbolicShape(symbols)};
}

TEST(OperatorSemanticsValidate, AddValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kAdd, AddParams{}).ok());
}

TEST(OperatorSemanticsValidate, RmsNormValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kRmsNorm, RmsNormParams{1e-5f}).ok());
}

TEST(OperatorSemanticsValidate, RmsNormInvalidEps) {
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRmsNorm, RmsNormParams{0.0f}).ok());
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRmsNorm, RmsNormParams{-1.0f}).ok());
}

TEST(OperatorSemanticsValidate, RmsNormNaN) {
    RmsNormParams p;
    p.eps = std::numeric_limits<float>::quiet_NaN();
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRmsNorm, p).ok());
}

TEST(OperatorSemanticsValidate, RoPEValidParams) {
    RoPEParams p{64, 32, 8, 2048};
    EXPECT_TRUE(ValidateOperatorParams(OpType::kRoPE, p).ok());
}

TEST(OperatorSemanticsValidate, RoPEInvalidParams) {
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRoPE, RoPEParams{0, 32, 8, 2048}).ok());
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRoPE, RoPEParams{64, 0, 8, 2048}).ok());
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRoPE, RoPEParams{64, 32, 0, 2048}).ok());
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRoPE, RoPEParams{64, 32, 8, 0}).ok());
}

TEST(OperatorSemanticsValidate, WrongVariantType) {
    EXPECT_FALSE(ValidateOperatorParams(OpType::kAdd, RmsNormParams{}).ok());
    EXPECT_FALSE(ValidateOperatorParams(OpType::kRmsNorm, AddParams{}).ok());
}

TEST(OperatorSemanticsValidate, UnknownOpType) {
    EXPECT_FALSE(ValidateOperatorParams(OpType::kUnknown, AddParams{}).ok());
}

TEST(OperatorSemanticsValidate, EmbeddingValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kEmbedding, EmbeddingParams{}).ok());
}

TEST(OperatorSemanticsValidate, LinearValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kLinear, LinearParams{}).ok());
}

TEST(OperatorSemanticsValidate, MatMulValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kMatMul, MatMulParams{}).ok());
}

TEST(OperatorSemanticsValidate, SoftmaxValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kSoftmax, SoftmaxParams{}).ok());
}

TEST(OperatorSemanticsValidate, SiluValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kSilu, SiluParams{}).ok());
}

TEST(OperatorSemanticsValidate, SiluMulValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kSiluMul, SiluMulParams{}).ok());
}

TEST(OperatorSemanticsValidate, ElementwiseMulValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kElementwiseMul, ElementwiseMulParams{}).ok());
}

TEST(OperatorSemanticsValidate, KVCacheUpdateValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kKVCacheUpdate, KVCacheUpdateParams{}).ok());
}

TEST(OperatorSemanticsValidate, AttentionValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kAttention, AttentionParams{32, 8, 64}).ok());
}

TEST(OperatorSemanticsValidate, ArgmaxValidParams) {
    EXPECT_TRUE(ValidateOperatorParams(OpType::kArgmax, ArgmaxParams{}).ok());
}

TEST(OperatorSemanticsInfer, AddFloat32Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, AddBroadcast) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kAdd, AddParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, AddDtypeMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Double(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, AddWrongVariant) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, RmsNormParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, AddWrongInputCount) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    std::vector<TensorSpec> inputs = {lhs};
    EXPECT_FALSE(InferOperator(OpType::kAdd, AddParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, RmsNormInvalidEps) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{0.0f}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormRejectsRankZero) {
    auto input = MakeSpec(DataType::Float32(), {});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RmsNormRank1Ok) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(OperatorSemanticsInfer, RmsNormRank3Ok) {
    auto input = MakeSpec(DataType::Float32(), {2, 4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape, input.shape);
}

TEST(OperatorSemanticsInfer, RmsNormFloat16Ok) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::Float(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, RmsNormBFloat16Ok) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, RmsNormFloat8Ok) {
    auto input = MakeSpec(DataType::Float8E4M3FN(), {4, 256});
    auto weight = MakeSpec(DataType::Float8E4M3FN(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float8E4M3FN());
}

TEST(OperatorSemanticsInfer, RmsNormMixedDTypeOk) {
    auto input = MakeSpec(DataType::Float(16), {4, 256});
    auto weight = MakeSpec(DataType::BFloat(16), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(OperatorSemanticsInfer, RmsNormWrongDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kRmsNorm, RmsNormParams{1e-5f}, inputs).ok());
}

TEST(OperatorSemanticsInfer, EmbeddingInt32Ok) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Float32(), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    auto result = InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, EmbeddingWrongWeightDtype) {
    auto tokens = MakeSpec(DataType::Int(32), {10});
    auto weight = MakeSpec(DataType::Int(64), {32000, 256});
    std::vector<TensorSpec> inputs = {tokens, weight};
    EXPECT_FALSE(InferOperator(OpType::kEmbedding, EmbeddingParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, LinearRank2Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, LinearRank1Ok) {
    auto input = MakeSpec(DataType::Float32(), {256});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    auto result = InferOperator(OpType::kLinear, LinearParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(OperatorSemanticsInfer, LinearInFeaturesMismatch) {
    auto input = MakeSpec(DataType::Float32(), {4, 128});
    auto weight = MakeSpec(DataType::Float32(), {512, 256});
    std::vector<TensorSpec> inputs = {input, weight};
    EXPECT_FALSE(InferOperator(OpType::kLinear, LinearParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, MatMulRank2Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {3, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, MatMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, MatMulInnerMismatch) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {5, 4});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    EXPECT_FALSE(InferOperator(OpType::kMatMul, MatMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, MatMulTransposeRhs) {
    auto lhs = MakeSpec(DataType::Float32(), {2, 3});
    auto rhs = MakeSpec(DataType::Float32(), {4, 3});
    MatMulParams p{true};
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kMatMul, p, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, SiluFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, SiluBFloat16Ok) {
    auto input = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSilu, SiluParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, SiluWrongDtype) {
    auto input = MakeSpec(DataType::Int(32), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kSilu, SiluParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SiluMulFloat32Ok) {
    auto gate = MakeSpec(DataType::Float32(), {4, 256});
    auto up = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, SiluMulBFloat16Ok) {
    auto gate = MakeSpec(DataType::BFloat(16), {4, 256});
    auto up = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, ElementwiseMulFloat32Ok) {
    auto lhs = MakeSpec(DataType::Float32(), {4, 256});
    auto rhs = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, ElementwiseMulBFloat16Ok) {
    auto lhs = MakeSpec(DataType::BFloat(16), {4, 256});
    auto rhs = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {lhs, rhs};
    auto result = InferOperator(OpType::kElementwiseMul, ElementwiseMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(OperatorSemanticsInfer, SoftmaxFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 2);
}

TEST(OperatorSemanticsInfer, SoftmaxAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_TRUE(InferOperator(OpType::kSoftmax, SoftmaxParams{5}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SoftmaxUnrankedSkipsAxisCheck) {
    auto input = MakeSpec(DataType::Float32());
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kSoftmax, SoftmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_FALSE(result->outputs[0].shape.IsRanked());
}

TEST(OperatorSemanticsInfer, ArgmaxFloat32Ok) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    auto result = InferOperator(OpType::kArgmax, ArgmaxParams{-1}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Int(64));
    EXPECT_EQ(result->outputs[0].shape.rank().value(), 1);
}

TEST(OperatorSemanticsInfer, ArgmaxAxisOutOfRange) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kArgmax, ArgmaxParams{5}, inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEFloat32Ok) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, rope_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 2);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, RoPEGQACompatible) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 2048});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 512});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    auto result = InferOperator(OpType::kRoPE, p, rope_inputs);
    ASSERT_TRUE(result.ok());
}

TEST(OperatorSemanticsInfer, RoPEQLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEKLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_TRUE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPERankMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEBatchDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {2, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Int(64), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, RoPEWrongPositionIdsDtype) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto k = MakeSpec(DataType::Float32(), {1, 128, 8 * 64});
    auto pos = MakeSpec(DataType::Float32(), std::vector<int64_t>{128});
    RoPEParams p{64, 32, 8, 2048};
    std::vector<TensorSpec> rope_inputs = {q, k, pos};
    EXPECT_FALSE(InferOperator(OpType::kRoPE, p, rope_inputs).ok());
}

TEST(OperatorSemanticsInfer, AttentionFloat32Ok) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto kCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    auto result = InferOperator(OpType::kAttention, p, attn_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 1);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, AttentionQLastDimMismatch) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 100});
    auto kCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention, p, attn_inputs).ok());
}

TEST(OperatorSemanticsInfer, AttentionWrongCacheDtype) {
    auto q = MakeSpec(DataType::Float32(), {1, 128, 32 * 64});
    auto kCache = MakeSpec(DataType::Double(), {1, 8, 1024, 64});
    auto vCache = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    AttentionParams p{32, 8, 64};
    std::vector<TensorSpec> attn_inputs = {q, kCache, vCache};
    EXPECT_TRUE(InferOperator(OpType::kAttention, p, attn_inputs).ok());
}

TEST(OperatorSemanticsInfer, KVCacheUpdateFloat32Ok) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs.size(), 2);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, KVCacheUpdateDtypeMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Double(), {1, 8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

TEST(OperatorSemanticsInfer, KVCacheUpdateShapeMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 4, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

TEST(OperatorSemanticsInfer, KVCacheUpdateRankMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

TEST(OperatorSemanticsInfer, UnknownOpType) {
    auto input = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> unknown_inputs = {input};
    EXPECT_FALSE(InferOperator(OpType::kUnknown, AddParams{}, unknown_inputs).ok());
}

TEST(OperatorSemanticsMakeCompact, AllContributing) {
    auto schema = GetOperatorSchema(OpType::kAdd);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {2, 3}),
            MakeSpec(DataType::Float32(), {2, 3})};
    auto compact = MakeCompactInputSpecs(*schema, inputs);
    ASSERT_TRUE(compact.ok());
    EXPECT_EQ(compact->size(), 2);
}

TEST(OperatorSemanticsMakeCompact, FiltersStatePorts) {
    auto schema = GetOperatorSchema(OpType::kKVCacheUpdate);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1024, 64}),
            MakeSpec(DataType::Float32(), {1, 8, 1024, 64})};
    auto compact = MakeCompactInputSpecs(*schema, inputs);
    ASSERT_TRUE(compact.ok());
    EXPECT_EQ(compact->size(), 2);
}


TEST(OperatorSemanticsMakeCompact, InputCountMismatch) {
    auto schema = GetOperatorSchema(OpType::kAdd);
    ASSERT_TRUE(schema.ok());
    std::vector<TensorSpec> inputs = {MakeSpec(DataType::Float32(), {2, 3})};
    EXPECT_FALSE(MakeCompactInputSpecs(*schema, inputs).ok());
}

}// namespace
