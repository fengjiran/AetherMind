#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(KVCacheUpdateInference, AcceptsValidParams) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, PreservesFloat32OutputDType) {
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

TEST(KVCacheUpdateInference, AcceptsDtypeMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Double(), {1, 8, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

TEST(KVCacheUpdateInference, AcceptsShapeMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 4, 1024, 64});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

TEST(KVCacheUpdateInference, AcceptsRankMismatch) {
    auto k = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto v = MakeSpec(DataType::Float32(), {1, 8, 1, 64});
    auto kCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024});
    auto vCacheIn = MakeSpec(DataType::Float32(), {1, 8, 1024, 64});
    std::vector<TensorSpec> kv_inputs = {k, v, kCacheIn, vCacheIn};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, kv_inputs).ok());
}

}// namespace
