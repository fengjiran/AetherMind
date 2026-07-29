#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "test_operator_inference_helpers.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// KVCacheUpdate cache tensors follow the rank-3 contract produced by
// KVCacheTensorSpec in graph_builder.cpp: {num_kv_heads, cache_len, head_dim}.
// The k/v activations for a single decoding step use {num_kv_heads, 1, head_dim}.
// Tests below use num_kv_heads=8, cache_len=1024, head_dim=64.

TEST(KVCacheUpdateInference, AcceptsValidParams) {
    auto k = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, PreservesRank3CacheShapeAndFloat32DType) {
    auto k = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs);
    ASSERT_TRUE(result.ok());
    ASSERT_EQ(result->outputs.size(), 2u);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
    ASSERT_EQ(result->outputs[0].shape.rank(), 3u);
    ASSERT_EQ(result->outputs[1].shape.rank(), 3u);
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 8);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 1024);
    EXPECT_EQ(result->outputs[0].shape[2].GetStaticValue(), 64);
}

TEST(KVCacheUpdateInference, AcceptsDtypeMismatchBetweenKVAndCache) {
    auto k = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Double(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Double());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(KVCacheUpdateInference, AcceptsFloat16KV) {
    auto k = MakeSpec(DataType::Float(16), {8, 1, 64});
    auto v = MakeSpec(DataType::Float(16), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(KVCacheUpdateInference, AcceptsBFloat16KV) {
    auto k = MakeSpec(DataType::BFloat(16), {8, 1, 64});
    auto v = MakeSpec(DataType::BFloat(16), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
}

TEST(KVCacheUpdateInference, RejectsInt32K) {
    auto k = MakeSpec(DataType::Int(32), {8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsFloat8E4M3FNKV) {
    auto k = MakeSpec(DataType::Float8E4M3FN(), {8, 1, 64});
    auto v = MakeSpec(DataType::Float8E4M3FN(), {8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

}// namespace
