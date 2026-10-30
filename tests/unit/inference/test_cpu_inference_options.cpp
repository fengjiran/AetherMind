#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/inference/cpu_inference_options.h"
#include "aethermind/runtime/runtime_builder.h"
#include "model/test_llama_checkpoint_helpers.h"
#include "test_utils/test_resource_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>

namespace {
using namespace aethermind;

HfModelConfig Configuration() {
    return test::MakeTinyLlamaConfig(2, false);
}

TEST(CpuRuntimeAssembly, MapsGeometryAndSelectsOnlyCpuFp32Resources) {
    const HfModelConfig config = Configuration();
    const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_TRUE(mapped->allocator.enable_cpu);
    EXPECT_FALSE(mapped->allocator.enable_cuda);
    EXPECT_FALSE(mapped->allocator.enable_cann);
    EXPECT_TRUE(mapped->backend.enable_cpu);
    EXPECT_FALSE(mapped->backend.enable_cuda);
    EXPECT_FALSE(mapped->backend.enable_cann);
    EXPECT_TRUE(mapped->execution.enable_default_stream);
    EXPECT_FALSE(mapped->execution.enable_graph_executor);
    EXPECT_FALSE(mapped->tracing.enable_profiling);
    EXPECT_FALSE(mapped->tracing.enable_memory_profiling);
    EXPECT_TRUE(mapped->kv_cache.enable_manager);
    EXPECT_EQ(mapped->kv_cache.num_layers, 2U);
    EXPECT_EQ(mapped->kv_cache.num_kv_heads, 2U);
    EXPECT_EQ(mapped->kv_cache.head_dim, 2U);
    EXPECT_EQ(mapped->kv_cache.max_tokens, 8U);
    EXPECT_EQ(mapped->kv_cache.kv_dtype, DataType::Float32());
    EXPECT_EQ(mapped->kv_cache.alignment, 64U);
    EXPECT_EQ(mapped->kv_cache.max_bytes, 0U);
    EXPECT_FALSE(mapped->workspace.enable_workspace_manager);
    EXPECT_EQ(mapped->workspace.default_workspace_limit_bytes, 0U);
}

TEST(CpuRuntimeAssembly, UsesPositiveExplicitHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = 4;
    config.hidden_size = config.num_attention_heads * config.head_dim;
    const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.head_dim, 4U);
}

TEST(CpuRuntimeAssembly, DerivesAbsentHeadDimensionAfterExactDivision) {
    HfModelConfig config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 12;
    const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.head_dim, 3U);
}

TEST(CpuRuntimeAssembly, RejectsNonintegralDerivedHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = 0;
    config.hidden_size = 9;
    const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
    EXPECT_FALSE(mapped.ok());
}

TEST(CpuRuntimeAssembly, RejectsNonpositiveDimensionsBeforeUnsignedConversion) {
    constexpr std::array<int64_t HfModelConfig::*, 2> fields{
            &HfModelConfig::num_hidden_layers, &HfModelConfig::num_key_value_heads};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            SCOPED_TRACE(value);
            HfModelConfig config = Configuration();
            config.*field = value;
            const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
            EXPECT_FALSE(mapped.ok());
        }
    }
}

TEST(CpuRuntimeAssembly, RejectsNegativeExplicitHeadDimension) {
    HfModelConfig config = Configuration();
    config.head_dim = -1;
    const auto mapped = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
    EXPECT_FALSE(mapped.ok());
}

TEST(CpuRuntimeAssembly, RejectsInvalidOperandsWhenDerivingHeadDimension) {
    constexpr std::array<int64_t HfModelConfig::*, 2> fields{
            &HfModelConfig::hidden_size, &HfModelConfig::num_attention_heads};
    for (const auto field: fields) {
        for (const int64_t value: {int64_t{0}, int64_t{-1}}) {
            SCOPED_TRACE(value);
            HfModelConfig config = Configuration();
            config.head_dim = 0;
            config.*field = value;
            EXPECT_FALSE(MakeCpuRuntimeOptions(config, {.context_tokens = 8}).ok());
        }
    }
}

TEST(CpuRuntimeAssembly, RejectsInvalidCapacity) {
    const HfModelConfig config = Configuration();
    EXPECT_FALSE(MakeCpuRuntimeOptions(config, {.context_tokens = 0}).ok());
    EXPECT_FALSE(MakeCpuRuntimeOptions(config, {.context_tokens = static_cast<size_t>(config.max_position_embeddings) + 1}).ok());
}

TEST(CpuRuntimeAssembly, ReturnsValuesIndependentOfSourceConfigLifetime) {
    const auto mapped = [] {
        HfModelConfig config = Configuration();
        auto result = MakeCpuRuntimeOptions(config, {.context_tokens = 8});
        config.num_hidden_layers = 99;
        config.num_key_value_heads = 99;
        config.head_dim = 99;
        return result;
    }();
    ASSERT_TRUE(mapped.ok()) << mapped.status().ToString();
    EXPECT_EQ(mapped->kv_cache.num_layers, 2U);
    EXPECT_EQ(mapped->kv_cache.num_kv_heads, 2U);
    EXPECT_EQ(mapped->kv_cache.head_dim, 2U);
}


TEST(CpuRuntimeConfiguration, RetainsUserSettingsAfterTheirLifetime) {
    const auto resolved = [] {
        auto config = Configuration();
        CpuRuntimeAssemblyOptions options{.context_tokens = 8,
                                          .kv_max_bytes = 4096,
                                          .workspace_limit_bytes = 128};
        options.cpu_feature_policy.disabled_features.Enable(CpuFeature::kAvx2);
        options.cpu_feature_policy.required_features.Enable(CpuFeature::kNeon);
        auto mapped = MakeCpuRuntimeOptions(config, options);
        options = {};
        config.num_hidden_layers = 99;
        return mapped;
    }();
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ(resolved->kv_cache.num_layers, 2U);
    EXPECT_EQ(resolved->kv_cache.max_bytes, 4096U);
    EXPECT_EQ(resolved->workspace.default_workspace_limit_bytes, 128U);
    EXPECT_TRUE(resolved->backend.cpu_feature_policy.disabled_features.Contains(CpuFeature::kAvx2));
    EXPECT_TRUE(resolved->backend.cpu_feature_policy.required_features.Contains(CpuFeature::kNeon));
}

TEST(CpuRuntimeConfiguration, ZeroByteCapsRemainUnlimited) {
    auto resolved = MakeCpuRuntimeOptions(Configuration(), {.context_tokens = 8});
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ(resolved->kv_cache.max_bytes, 0U);
    EXPECT_EQ(resolved->workspace.default_workspace_limit_bytes, 0U);
    RuntimeBuilder builder;
    builder.WithOptions(*resolved);
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    ASSERT_NE(runtime->GetKVCacheManager(), nullptr);
    EXPECT_EQ(runtime->GetKVCacheManager()->total_bytes(), 4096U);
}

TEST(CpuRuntimeConfiguration, KvCapRejectsBeforePayloadAndCanRetry) {
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    RuntimeBuilder builder;
    builder.RegisterCustomAllocatorProvider(kCPU, std::make_unique<test::ResourceRecordingProvider>(stats));
    auto resolved = MakeCpuRuntimeOptions(Configuration(), {.context_tokens = 8, .kv_max_bytes = 4095});
    ASSERT_TRUE(resolved.ok());
    builder.WithOptions(*resolved);
    EXPECT_EQ(builder.TryBuild().status().code(), StatusCode::kResourceExhausted);
    EXPECT_TRUE(stats->requests.empty());

    // A resource-admission failure retains the existing registration-transfer
    // contract: re-register the provider after the discarded registry.
    builder.RegisterCustomAllocatorProvider(kCPU, std::make_unique<test::ResourceRecordingProvider>(stats));
    resolved = MakeCpuRuntimeOptions(Configuration(), {.context_tokens = 8, .kv_max_bytes = 4096});
    ASSERT_TRUE(resolved.ok());
    builder.WithOptions(*resolved);
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    ASSERT_NE(runtime->GetKVCacheManager(), nullptr);
    EXPECT_EQ(runtime->GetKVCacheManager()->total_bytes(), 4096U);
    EXPECT_EQ(stats->requests.size(), 2U);
}

TEST(CpuRuntimeConfiguration, FeaturePolicyReachesTheCpuBackend) {
    CpuRuntimeAssemblyOptions options{.context_tokens = 8};
    options.cpu_feature_policy.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma});
    auto resolved = MakeCpuRuntimeOptions(Configuration(), options);
    ASSERT_TRUE(resolved.ok());
    RuntimeBuilder builder;
    builder.WithOptions(*resolved);
    resolved->backend.cpu_feature_policy = {};
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    auto backend = runtime->GetBackend(kCPU);
    ASSERT_TRUE(backend.ok()) << backend.status().ToString();
    const auto* cpu = dynamic_cast<const CpuBackend*>(*backend);
    ASSERT_NE(cpu, nullptr);
    EXPECT_FALSE(cpu->cpu_capabilities().effective_features.Contains(CpuFeature::kAvx2));
    EXPECT_FALSE(cpu->cpu_capabilities().effective_features.Contains(CpuFeature::kFma));
}

TEST(CpuRuntimeConfiguration, BuilderUsesAnOwningConfigurationSnapshot) {
    auto resolved = MakeCpuRuntimeOptions(Configuration(), {.context_tokens = 8,
                                                            .kv_max_bytes = 4096,
                                                            .workspace_limit_bytes = 128});
    ASSERT_TRUE(resolved.ok());
    RuntimeBuilder builder;
    builder.WithOptions(*resolved);
    resolved->kv_cache = {};
    resolved->workspace.default_workspace_limit_bytes = 1;
    auto runtime = builder.TryBuild();
    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    ASSERT_NE(runtime->GetKVCacheManager(), nullptr);
    EXPECT_EQ(runtime->GetKVCacheManager()->capacity_tokens(), 8U);
    EXPECT_EQ(runtime->workspace_limit_bytes(), 128U);
}

} // namespace
