#include "aethermind/inference/executable_model.h"
#include "aethermind/inference/inference_options.h"
#include "aethermind/inference/inference_session.h"
#include "aethermind/runtime/runtime_builder.h"
#include "inference/test_reduced_weight_inference_helpers.h"

#include <array>
#include <gtest/gtest.h>
#include <memory>

namespace {

using namespace aethermind;
using namespace aethermind::test::reduced;

// Step 8 verifies the explicit kv_dtype config chain: it must drive the physical
// Runtime KV layout and the graph state specs, close the Load/Generate loop for
// the retained-W16 policies, and reject inconsistent requests early. Numerical
// quality across the four W x KV combinations is accepted separately (step 9).

Runtime MakeKvRuntime(int64_t kv_heads, DataType kv_dtype) {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 2,
                        .num_kv_heads = static_cast<size_t>(kv_heads),
                        .max_tokens = 8,
                        .head_dim = 2,
                        .kv_dtype = kv_dtype,
                        .alignment = 64};
    RuntimeBuilder builder;
    builder.WithOptions(options);
    return builder.Build();
}

ModelLoadOptions W16KvOptions(bool bfloat16, DataType kv_dtype) {
    return {.preset = CompilePreset::kO1Plain,
            .weights = bfloat16 ? WeightPolicy::kKeepBf16WeightsFp32Activation
                                : WeightPolicy::kKeepFp16WeightsFp32Activation,
            .kv_dtype = kv_dtype};
}

TEST(KV16LoadChain, LoadAndGenerateCloseTheLoopForBothKvDTypes) {
    for (const bool bfloat16: {false, true}) {
        const DataType kv_dtype = bfloat16 ? DataType::BFloat(16) : DataType::Float(16);
        SCOPED_TRACE(ToString(kv_dtype));
        const CompatibilityCase test_case{bfloat16, false, 4, "KV16LoadChain", "float32",
                                          DataType::Float32()};
        TempDirectory directory;
        const auto pair = WriteCheckpointPair(directory.path(), test_case);
        Runtime runtime = MakeKvRuntime(test_case.kv_heads, kv_dtype);
        ASSERT_NE(runtime.GetKVCacheManager(), nullptr);
        EXPECT_EQ(runtime.GetKVCacheManager()->layout().kv_dtype, kv_dtype);

        auto model = ExecutableModel::Load(runtime, pair.source, W16KvOptions(bfloat16, kv_dtype));
        ASSERT_TRUE(model.ok()) << model.status().ToString();

        auto session = InferenceSession::Create(
                runtime, std::make_shared<ExecutableModel>(std::move(*model)));
        ASSERT_TRUE(session.ok()) << session.status().ToString();
        constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
        const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
        ASSERT_TRUE(generated.ok()) << generated.status().ToString();
        EXPECT_EQ(generated->size(), 3U);
        const auto repeated = session->Generate(prompt, {.max_new_tokens = 3});
        ASSERT_TRUE(repeated.ok()) << repeated.status().ToString();
        EXPECT_EQ(*repeated, *generated);
    }
}

TEST(KV16LoadChain, Kv16WeightDtypeIndependentOfKvDtype) {
    // F16 matrices with a BF16 cache: W and KV precision are independent.
    const CompatibilityCase test_case{false, false, 4, "KV16Cross", "float32", DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeKvRuntime(test_case.kv_heads, DataType::BFloat(16));
    auto model = ExecutableModel::Load(runtime, pair.source,
                                       W16KvOptions(false, DataType::BFloat(16)));
    ASSERT_TRUE(model.ok()) << model.status().ToString();
    auto session = InferenceSession::Create(
            runtime, std::make_shared<ExecutableModel>(std::move(*model)));
    ASSERT_TRUE(session.ok()) << session.status().ToString();
    constexpr std::array<uint32_t, 3> prompt{1, 7, 3};
    const auto generated = session->Generate(prompt, {.max_new_tokens = 3});
    EXPECT_TRUE(generated.ok()) << generated.status().ToString();
}

TEST(KV16LoadChain, RejectsRuntimeKvDTypeMismatch) {
    const CompatibilityCase test_case{false, false, 4, "KV16Mismatch", "float32",
                                      DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeKvRuntime(test_case.kv_heads, DataType::Float(16));
    // Runtime is F16 but the request asks for BF16: must fail before conversion.
    const auto model = ExecutableModel::Load(runtime, pair.source,
                                             W16KvOptions(false, DataType::BFloat(16)));
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status(), StatusCode::kInvalidArgument);
}

TEST(KV16LoadChain, RejectsKv16WithoutRetainedW16Policy) {
    const CompatibilityCase test_case{false, false, 4, "KV16Policy", "float32",
                                      DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeKvRuntime(test_case.kv_heads, DataType::Float(16));
    ModelLoadOptions options{.preset = CompilePreset::kO1Plain,
                             .weights = WeightPolicy::kRequireFp32,
                             .kv_dtype = DataType::Float(16)};
    const auto model = ExecutableModel::Load(runtime, pair.source, options);
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
}

TEST(KV16LoadChain, RejectsInvalidKvDType) {
    const CompatibilityCase test_case{false, false, 4, "KV16Invalid", "float32",
                                      DataType::Float32()};
    TempDirectory directory;
    const auto pair = WriteCheckpointPair(directory.path(), test_case);
    Runtime runtime = MakeKvRuntime(test_case.kv_heads, DataType::Float32());
    ModelLoadOptions options = W16KvOptions(false, DataType::Int(32));
    const auto model = ExecutableModel::Load(runtime, pair.source, options);
    ASSERT_FALSE(model.ok());
    EXPECT_EQ(model.status(), StatusCode::kInvalidArgument);
}

TEST(KV16LoadChain, MakeKVCacheOptionsRejectsNonScalarDTypes) {
    HfModelConfig config;
    config.num_hidden_layers = 2;
    config.num_key_value_heads = 4;
    config.num_attention_heads = 4;
    config.hidden_size = 8;
    config.head_dim = 2;
    config.max_position_embeddings = 32;
    EXPECT_TRUE(MakeKVCacheOptions(config, 8, DataType::Float32()).ok());
    EXPECT_TRUE(MakeKVCacheOptions(config, 8, DataType::Float(16)).ok());
    EXPECT_TRUE(MakeKVCacheOptions(config, 8, DataType::BFloat(16)).ok());
    EXPECT_EQ(MakeKVCacheOptions(config, 8, DataType{}).status(), StatusCode::kInvalidArgument);
    EXPECT_EQ(MakeKVCacheOptions(config, 8, DataType::ScalableFloat(32, 2)).status(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(MakeKVCacheOptions(config, 8, DataType::Int(32)).status(),
              StatusCode::kInvalidArgument);
    // The FP32 wrapper preserves the existing default behavior.
    const auto wrapped = MakeFp32KVCacheOptions(config, 8);
    ASSERT_TRUE(wrapped.ok());
    EXPECT_EQ(wrapped->kv_dtype, DataType::Float32());
}

} // namespace