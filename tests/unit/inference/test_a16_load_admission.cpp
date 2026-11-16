#include "aethermind/inference/executable_model.h"
#include "aethermind/runtime/runtime_builder.h"

#include <filesystem>
#include <gtest/gtest.h>
#include <string>

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
    Runtime runtime = MakeRuntime();
    const ModelLoadOptions options{
            .weights = WeightPolicy::kKeepBf16WeightsFp32Activation,
            .activation_dtype = DataType::BFloat(16)};
    const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
    EXPECT_FALSE(model.ok());
    EXPECT_NE(model.status(), StatusCode::kUnimplemented);
}

TEST(A16LoadAdmission, RejectsFloat16ActivationUntilP4bResidualCastsExist) {
    // P4-A executes the full-same-type BF16 preset only; FP16 keeps an F32
    // residual stream and needs Cast kernels on that boundary (P4-B).
    Runtime runtime = MakeRuntime();
    const ModelLoadOptions options{
            .weights = WeightPolicy::kKeepFp16WeightsFp32Activation,
            .activation_dtype = DataType::Float(16)};
    const auto model = ExecutableModel::Load(runtime, kMissingDirectory, options);
    EXPECT_EQ(model.status(), StatusCode::kUnimplemented);
    EXPECT_NE(model.status().message().find("P4-B"), std::string::npos)
            << model.status().message();
}

} // namespace
