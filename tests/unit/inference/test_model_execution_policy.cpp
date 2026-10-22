#include "aethermind/inference/model_execution_policy.h"

#include "aethermind/model/formats/hf/hf_model_config.h"
#include "model/test_llama_checkpoint_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>

namespace {

using namespace aethermind;

HfModelConfig Configuration() {
    return test::MakeTinyLlamaConfig(2, false);
}

TEST(Fp32KVCacheOptions, MapsLogicalGeometryWithoutAllocatingStorage) {
    const auto mapped = MakeFp32KVCacheOptions(Configuration(), 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_TRUE(mapped->enable_manager);
    EXPECT_EQ(mapped->num_layers, 2U);
    EXPECT_EQ(mapped->num_kv_heads, 2U);
    EXPECT_EQ(mapped->head_dim, 2U);
    EXPECT_EQ(mapped->max_tokens, 8U);
    EXPECT_EQ(mapped->kv_dtype, DataType::Float32());
    EXPECT_EQ(mapped->alignment, 64U);
}

TEST(Fp32KVCacheOptions, UsesExplicitLogicalHeadDimension) {
    auto config = Configuration();
    config.head_dim = 4;
    config.hidden_size = config.num_attention_heads * config.head_dim;
    const auto mapped = MakeFp32KVCacheOptions(config, 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->head_dim, 4U);
}

TEST(Fp32KVCacheOptions, DerivesAbsentHeadDimensionWithExactDivision) {
    auto config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 12;
    const auto mapped = MakeFp32KVCacheOptions(config, 8);
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->head_dim, 3U);
}

TEST(Fp32KVCacheOptions, RejectsNegativeExplicitHeadDimensionInsteadOfFallingBack) {
    auto config = Configuration();
    config.head_dim = -1;
    const auto mapped = MakeFp32KVCacheOptions(config, 8);
    EXPECT_EQ(mapped.status().code(), StatusCode::kInvalidArgument);
}

TEST(Fp32KVCacheOptions, RejectsNonintegralDerivedHeadDimension) {
    auto config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 9;
    const auto mapped = MakeFp32KVCacheOptions(config, 8);
    EXPECT_EQ(mapped.status().code(), StatusCode::kInvalidArgument);
}

TEST(Fp32KVCacheOptions, RejectsNonpositiveGeometryBeforeUnsignedConversion) {
    constexpr std::array<int64_t HfModelConfig::*, 3> fields{
            &HfModelConfig::num_hidden_layers,
            &HfModelConfig::num_key_value_heads,
            &HfModelConfig::max_position_embeddings};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            auto config = Configuration();
            config.*field = value;
            const auto mapped = MakeFp32KVCacheOptions(config, 8);
            EXPECT_EQ(mapped.status().code(), StatusCode::kInvalidArgument);
        }
    }
}

TEST(Fp32KVCacheOptions, RejectsNonpositiveHeadDimensionDivisionOperands) {
    constexpr std::array<int64_t HfModelConfig::*, 2> fields{
            &HfModelConfig::hidden_size,
            &HfModelConfig::num_attention_heads};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            auto config = Configuration();
            config.head_dim = 0;
            config.*field = value;
            const auto mapped = MakeFp32KVCacheOptions(config, 8);
            EXPECT_EQ(mapped.status().code(), StatusCode::kInvalidArgument);
        }
    }
}

TEST(Fp32KVCacheOptions, AcceptsCapacityBelowOrAtModelContext) {
    const auto config = Configuration();
    for (const size_t capacity: {size_t{1}, static_cast<size_t>(config.max_position_embeddings)}) {
        const auto mapped = MakeFp32KVCacheOptions(config, capacity);
        ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
        EXPECT_EQ(mapped->max_tokens, capacity);
    }
}

TEST(Fp32KVCacheOptions, RejectsZeroCapacityAndCapacityBeyondModelContext) {
    const auto config = Configuration();
    const auto zero = MakeFp32KVCacheOptions(config, 0);
    EXPECT_EQ(zero.status().code(), StatusCode::kInvalidArgument);
    const auto over = MakeFp32KVCacheOptions(config, static_cast<size_t>(config.max_position_embeddings) + 1);
    EXPECT_EQ(over.status().code(), StatusCode::kOutOfRange);
}

TEST(Fp32KVCacheOptions, ReturnsOwningValuesIndependentOfConfigLifetime) {
    const auto mapped = [] {
        auto config = Configuration();
        auto result = MakeFp32KVCacheOptions(config, 8);
        config.num_hidden_layers = 99;
        config.num_key_value_heads = 99;
        config.head_dim = 99;
        return result;
    }();
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->num_layers, 2U);
    EXPECT_EQ(mapped->num_kv_heads, 2U);
    EXPECT_EQ(mapped->head_dim, 2U);
}

TEST(Fp32KVCacheOptions, ChecksSizeRepresentabilityWithoutMaterializingPlanes) {
    auto config = Configuration();
    config.num_hidden_layers = std::numeric_limits<int64_t>::max();
    const auto mapped = MakeFp32KVCacheOptions(config, 8);
    if constexpr (sizeof(size_t) < sizeof(int64_t)) {
        EXPECT_EQ(mapped.status().code(), StatusCode::kOutOfRange);
    } else {
        ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
        EXPECT_EQ(mapped->num_layers, static_cast<size_t>(config.num_hidden_layers));
    }
}

} // namespace
