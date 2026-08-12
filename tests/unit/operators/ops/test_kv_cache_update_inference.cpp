#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// KVCacheUpdate contract (matches graph_builder.cpp):
//   k, v             : rank 2, shape [T, Hkv*D]
//   k_cache, v_cache : rank 3, shape [Hkv, C, D]
//   All four inputs share the same dtype in {Float32, Float16, BFloat16}.
// Tests use Hkv=8, D=64, C=1024, T=1 (so Hkv*D=512).

// Helper: build a consistent 4-input set with the given dtype and cache_len.
std::vector<TensorSpec> MakeConsistentInputs(DataType dtype, int64_t t = 1,
                                             int64_t kv_heads = 8,
                                             int64_t cache_len = 1024,
                                             int64_t head_dim = 64) {
    return {
            MakeSpec(dtype, {t, kv_heads * head_dim}),
            MakeSpec(dtype, {t, kv_heads * head_dim}),
            MakeSpec(dtype, {kv_heads, cache_len, head_dim}),
            MakeSpec(dtype, {kv_heads, cache_len, head_dim}),
    };
}

TEST(KVCacheUpdateInference, AcceptsValidParams) {
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{},
                              MakeConsistentInputs(DataType::Float32()))
                        .ok());
}

TEST(KVCacheUpdateInference, PreservesRank3CacheShapeAndDType) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    auto result = InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 2u);
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    EXPECT_EQ(result->outputs[1].dtype, DataType::Float32());
    ASSERT_EQ(result->outputs[0].shape.rank(), 3u);
    ASSERT_EQ(result->outputs[1].shape.rank(), 3u);
    EXPECT_EQ(result->outputs[0].shape[0].GetStaticValue(), 8);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 1024);
    EXPECT_EQ(result->outputs[0].shape[2].GetStaticValue(), 64);
}

TEST(KVCacheUpdateInference, AcceptsFloat16) {
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{},
                              MakeConsistentInputs(DataType::Float(16)))
                        .ok());
}

TEST(KVCacheUpdateInference, AcceptsBFloat16) {
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{},
                              MakeConsistentInputs(DataType::BFloat(16)))
                        .ok());
}

// --- Dtype rejection ---

TEST(KVCacheUpdateInference, RejectsInt32K) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    inputs[0] = {DataType::Int(32), inputs[0].shape};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsFloat8E4M3FNKV) {
    auto inputs = MakeConsistentInputs(DataType::Float8E4M3FN());
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsMismatchedKVAndCacheDtype) {
    // FP16 activations + FP32 cache is rejected: no implicit conversion.
    auto inputs = MakeConsistentInputs(DataType::Float32());
    inputs[0] = {DataType::Float(16), inputs[0].shape};
    inputs[1] = {DataType::Float(16), inputs[1].shape};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsMismatchedKAndVDtype) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    inputs[1] = {DataType::Float(16), inputs[1].shape};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

// --- Rank rejection ---

TEST(KVCacheUpdateInference, RejectsRank3KVActivation) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    // Wrong: k/v as rank 3 (e.g. [Hkv, T, D]) instead of rank 2 [T, Hkv*D].
    inputs[0] = MakeSpec(DataType::Float32(), {8, 1, 64});
    inputs[1] = MakeSpec(DataType::Float32(), {8, 1, 64});
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsRank2Cache) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    // Wrong: cache as rank 2 instead of rank 3.
    inputs[2] = MakeSpec(DataType::Float32(), {8, 512});
    inputs[3] = MakeSpec(DataType::Float32(), {8, 512});
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

// --- Shape mismatch rejection ---

TEST(KVCacheUpdateInference, RejectsKAndVShapeMismatch) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    // v has a different T than k.
    inputs[1] = MakeSpec(DataType::Float32(), {2, 512});
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsKCacheAndVCacheShapeMismatch) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    // v_cache has a different cache_len than k_cache.
    inputs[3] = MakeSpec(DataType::Float32(), {8, 2048, 64});
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsKLastDimNotMatchingHkvTimesD) {
    auto inputs = MakeConsistentInputs(DataType::Float32());
    // k last dim = 256 but Hkv*D = 8*64 = 512.
    inputs[0] = MakeSpec(DataType::Float32(), {1, 256});
    inputs[1] = MakeSpec(DataType::Float32(), {1, 256});
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

// --- Static positivity rejection ---

TEST(KVCacheUpdateInference, RejectsZeroSequenceLength) {
    auto inputs = MakeConsistentInputs(DataType::Float32(), /*t=*/0);
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsZeroNumKVHeads) {
    auto inputs = MakeConsistentInputs(DataType::Float32(), 1, /*kv_heads=*/0);
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsZeroHeadDim) {
    auto inputs = MakeConsistentInputs(
            DataType::Float32(), 1, 8, 1024, /*head_dim=*/0);
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsZeroCacheCapacity) {
    auto inputs = MakeConsistentInputs(DataType::Float32(), 1, 8, /*cache_len=*/0);
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsSequenceLengthExceedingCacheCapacity) {
    // Static T > static cache_len violates the graph-level capacity contract
    // (seq_len <= cache_len); prefill beyond capacity fails at build time.
    auto inputs = MakeConsistentInputs(DataType::Float32(), /*t=*/2048, 8, /*cache_len=*/1024);
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, RejectsZeroKLastDimWithSymbolicCache) {
    // Zero hidden dim must be rejected even when kv_heads/head_dim are
    // symbolic (the static hidden == kv_heads*head_dim check cannot fire).
    auto inputs = MakeConsistentInputs(DataType::Float32());
    inputs[0] = MakeSpec(DataType::Float32(), {1, 0});
    inputs[1] = MakeSpec(DataType::Float32(), {1, 0});
    const ShapeSymbol sym_heads = ShapeSymbol::Create();
    const ShapeSymbol sym_dim = ShapeSymbol::Create();
    const SymbolicShape sym_cache_shape{sym_heads, ShapeSymbol::CreateFromValue(1024),
                                        sym_dim};
    inputs[2] = {DataType::Float32(), sym_cache_shape};
    inputs[3] = {DataType::Float32(), sym_cache_shape};
    EXPECT_FALSE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

// --- Symbolic axis acceptance ---

TEST(KVCacheUpdateInference, AcceptsSymbolicCacheLen) {
    // Cache capacity is symbolic (deferred to runtime); must still pass.
    auto inputs = MakeConsistentInputs(DataType::Float32());
    const ShapeSymbol sym_cache = ShapeSymbol::Create();
    const SymbolicShape cache_shape{ShapeSymbol::CreateFromValue(8), sym_cache,
                                    ShapeSymbol::CreateFromValue(64)};
    inputs[2] = {DataType::Float32(), cache_shape};
    inputs[3] = {DataType::Float32(), cache_shape};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, AcceptsSymbolicSequenceLength) {
    // T is symbolic (deferred to runtime); must still pass.
    auto inputs = MakeConsistentInputs(DataType::Float32());
    const ShapeSymbol sym_t = ShapeSymbol::Create();
    const SymbolicShape kv_shape{sym_t, ShapeSymbol::CreateFromValue(512)};
    inputs[0] = {DataType::Float32(), kv_shape};
    inputs[1] = {DataType::Float32(), kv_shape};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

TEST(KVCacheUpdateInference, AcceptsSymbolicSequenceLengthWithStaticCache) {
    // Symbolic T cannot be proven <= cache_len at graph time; acceptance is
    // deferred to the runtime capacity check (KVCacheView::ValidateWrite).
    auto inputs = MakeConsistentInputs(DataType::Float32(), /*t=*/1024, 8, /*cache_len=*/512);
    const ShapeSymbol sym_t = ShapeSymbol::Create();
    const SymbolicShape kv_shape{sym_t, ShapeSymbol::CreateFromValue(512)};
    inputs[0] = {DataType::Float32(), kv_shape};
    inputs[1] = {DataType::Float32(), kv_shape};
    EXPECT_TRUE(InferOperator(OpType::kKVCacheUpdate, KVCacheUpdateParams{}, inputs).ok());
}

}// namespace
