#include "aethermind/inference/executable_model.h"
#include "aethermind/runtime/runtime_builder.h"

#include <filesystem>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// Admission runs before any checkpoint I/O, so a missing directory is enough to
// distinguish "rejected by the A16 contract" from "accepted and forwarded to
// the loader".

const std::filesystem::path kMissingDirectory = "/nonexistent/aethermind-a16-admission";

Runtime MakeRuntime() {
    RuntimeBuilder builder;
    return builder.Build();
}

TEST(A16LoadAdmission, RejectsCrossDtypeActivationAndWeightPolicy) {
    Runtime runtime = MakeRuntime();
    const ModelLoadOptions options{
            .weights = WeightPolicy::kKeepFp16WeightsFp32Activation,
            .activation_dtype = DataType::BFloat(16)};
    const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
    EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
}

TEST(A16LoadAdmission, RejectsActivationWithoutRetainedW16Policy) {
    Runtime runtime = MakeRuntime();
    for (const WeightPolicy policy: {WeightPolicy::kRequireFp32,
                                     WeightPolicy::kConvertToFp32}) {
        const ModelLoadOptions options{.weights = policy,
                                       .activation_dtype = DataType::BFloat(16)};
        const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
        EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
    }
}

TEST(A16LoadAdmission, RejectsNonScalarOrUnsupportedActivation) {
    Runtime runtime = MakeRuntime();
    for (const DataType dtype: {DataType{}, DataType::Int(32), DataType::Double()}) {
        const ModelLoadOptions options{
                .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
                .activation_dtype = dtype};
        const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
        EXPECT_EQ(model.status(), StatusCode::kInvalidArgument);
    }
}

TEST(A16LoadAdmission, RejectsNonCpuTarget) {
    Runtime runtime = MakeRuntime();
    const ModelLoadOptions options{
            .target = DeviceType::kCUDA,
            .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
            .activation_dtype = DataType::BFloat(16)};
    const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
    EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
}

TEST(A16LoadAdmission, AcceptsSameDtypePresetAndForwardsToLoader) {
    for (const auto dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        RuntimeOptions runtime_options;
        runtime_options.kv_cache = {.enable_manager = true, .num_layers = 1, .num_kv_heads = 1, .max_tokens = 2, .head_dim = 2, .kv_dtype = dtype};
        RuntimeBuilder builder;
        builder.WithOptions(runtime_options);
        Runtime runtime = builder.Build();
        const ModelLoadOptions options{
                .weights = dtype == DataType::Float(16) ? WeightPolicy::kKeepFp16WeightsFp32Activation
                                                        : WeightPolicy::kKeepBf16WeightsFp32Activation,
                .kv_dtype = dtype,
                .activation_dtype = dtype};
        const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
        ASSERT_FALSE(model.ok());
        EXPECT_TRUE(model.status().message().starts_with("ExecutableModel::Load loading failed:"))
                << model.status().ToString();
    }
}

TEST(A16LoadAdmission, RejectsUnmatchedKvDTypeBeforeModelIo) {
    Runtime runtime = MakeRuntime();
    for (const auto kv_dtype: {DataType::Float32(), DataType::BFloat(16)}) {
        const ModelLoadOptions options{
                .weights = WeightPolicy::kKeepFp16WeightsFp32Activation,
                .kv_dtype = kv_dtype,
                .activation_dtype = DataType::Float(16)};
        const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
        EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
        EXPECT_EQ(model.status().message(), "A16 activation requires the same-dtype KV storage");
    }
}

} // namespace
