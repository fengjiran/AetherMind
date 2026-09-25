#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/silu/silu_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

using namespace aethermind;
using cpu::detail::SiluF32KernelArgs;

KernelSelector MakeSelector(const DataType& dtype = DataType::Float32()) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = dtype,
            .weight_dtype = dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareSiluKernel(const DataType& dtype = DataType::Float32()) {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kSilu, MakeSelector(dtype), OpParams{SiluParams{}});
}

struct SiluTestViews {
    TensorView input{};
    MutableTensorView output{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildSiluPreparedParams(const ResolvedKernel& kernel,
                                                       const SiluTestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 1> inputs{views.input};
    const std::array<MutableTensorView, 1> outputs{views.output};
    const Status status = kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
            },
            prepared.storage.data());
    if (!status.ok()) {
        return status;
    }
    return prepared;
}

Status RunSilu(const SiluTestViews& views) noexcept {
    const auto kernel = PrepareSiluKernel();
    if (!kernel.ok()) {
        return kernel.status();
    }
    const auto prepared = BuildSiluPreparedParams(*kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return kernel->fn(KernelContext{
            .kernel_params = prepared->storage.data(),
            .attrs = kernel->attrs,
    });
}

float ExpectedSilu(float value) {
    if (value >= 0.0F) {
        return value / (1.0F + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return value * exp_value / (1.0F + exp_value);
}

TEST(CPUKernelSilu, ResolvesF32ReferenceDescriptor) {
    const auto kernel = PrepareSiluKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    EXPECT_EQ(kernel->name, "cpu::silu_f32_reference");
    EXPECT_EQ(kernel->op_type, OpType::kSilu);
    EXPECT_EQ(kernel->params_size, sizeof(SiluF32KernelArgs));
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_TRUE(kernel->attrs.empty());
    EXPECT_EQ(kernel->workspace_requirement.bytes, 0U);
}

TEST(CPUKernelSilu, Float16SelectorDoesNotResolve) {
    const auto kernel = PrepareSiluKernel(DataType::Float(16));
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kNotFound) << kernel.status().ToString();
}

TEST(CPUKernelSilu, ContiguousValuesUseStableLargeMagnitudeFormula) {
    constexpr std::array input{-100.0F, -2.0F, 0.0F, 2.0F, 100.0F};
    std::array<float, input.size()> output{};
    constexpr int64_t shape[1] = {static_cast<int64_t>(input.size())};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{input.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (size_t index = 0; index < input.size(); ++index) {
        EXPECT_FLOAT_EQ(output[index], ExpectedSilu(input[index]));
        EXPECT_TRUE(std::isfinite(output[index]));
    }
    EXPECT_LT(output.front(), 0.0F);
    EXPECT_NEAR(output[1], -0.23840584F, 1.0e-7F);
    EXPECT_NEAR(output[3], 1.76159418F, 2.0e-7F);
    EXPECT_NEAR(output.back(), 100.0F, 1.0e-5F);
}

TEST(CPUKernelSilu, StridedInputAndTransposedOutput) {
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_strides[2] = {1, 2};
    constexpr float sentinel = -1234.0F;
    const std::array input{-2.0F, -1.0F, 0.0F, sentinel, 1.0F, 2.0F, 3.0F};
    std::array<float, 6> output{};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{input.data(), DataType::Float32(), shape, input_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, output_strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (int64_t row = 0; row < shape[0]; ++row) {
        for (int64_t column = 0; column < shape[1]; ++column) {
            const int64_t input_offset = row * input_strides[0] + column * input_strides[1];
            const int64_t output_offset = row * output_strides[0] + column * output_strides[1];
            EXPECT_FLOAT_EQ(output[static_cast<size_t>(output_offset)],
                            ExpectedSilu(input[static_cast<size_t>(input_offset)]));
        }
    }
}

TEST(CPUKernelSilu, RankZeroScalar) {
    const float input = -1.5F;
    float output = 0.0F;

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{&input, DataType::Float32(), {}, {}},
            .output = MutableTensorView{&output, DataType::Float32(), {}, {}},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output, ExpectedSilu(input));
}

TEST(CPUKernelSilu, ZeroElementOutputNeedsNoStorage) {
    constexpr int64_t shape[2] = {2, 0};
    constexpr int64_t strides[2] = {0, 1};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{nullptr, DataType::Float32(), shape, strides},
            .output = MutableTensorView{nullptr, DataType::Float32(), shape, strides},
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelSilu, AcceptsExactStridedInPlace) {
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t strides[2] = {1, 2};
    std::array<float, 6> storage{-2.0F, 1.0F, -1.0F, 2.0F, 0.0F, 3.0F};
    const std::array<float, 6> original = storage;

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{storage.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{storage.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (int64_t row = 0; row < shape[0]; ++row) {
        for (int64_t column = 0; column < shape[1]; ++column) {
            const int64_t offset = row * strides[0] + column * strides[1];
            EXPECT_FLOAT_EQ(storage[static_cast<size_t>(offset)],
                            ExpectedSilu(original[static_cast<size_t>(offset)]));
        }
    }
}

TEST(CPUKernelSilu, RejectsPartialInputOutputOverlap) {
    std::array<float, 5> storage{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{storage.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{storage.data() + 1, DataType::Float32(), shape, strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSilu, RejectsNonInjectiveOutputMapping) {
    constexpr std::array input{1.0F, 2.0F, 3.0F, 4.0F};
    std::array<float, 4> output{};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t output_strides[2] = {1, 1};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{input.data(), DataType::Float32(), shape, input_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, output_strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSilu, RejectsOutputLayoutWhoseInjectivityCannotBeProven) {
    constexpr std::array input{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    std::array<float, 8> output{};
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t output_strides[2] = {3, 2};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{input.data(), DataType::Float32(), shape, input_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, output_strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelSilu, ReportsUndecidableStridedInputOutputOverlap) {
    std::array<float, 9> storage{};
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {2};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{storage.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{storage.data() + 1, DataType::Float32(), shape, strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelSilu, RejectsMismatchedShapesAndDtypes) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float output[4] = {};
    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t output_shape[1] = {4};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t output_strides[1] = {1};

    const Status shape_status = RunSilu(SiluTestViews{
            .input = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(shape_status.code(), StatusCode::kInvalidArgument) << shape_status.ToString();

    const Status dtype_status = RunSilu(SiluTestViews{
            .input = TensorView{input, DataType::Float32(), output_shape, output_strides},
            .output = MutableTensorView{output, DataType::Float(16), output_shape, output_strides},
    });
    EXPECT_EQ(dtype_status.code(), StatusCode::kInvalidArgument) << dtype_status.ToString();
}

TEST(CPUKernelSilu, RejectsAddressOverflowDuringBinding) {
    constexpr float input[1] = {1.0F};
    float output[1] = {};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t huge_strides[2] = {std::numeric_limits<int64_t>::max(), 1};

    const Status status = RunSilu(SiluTestViews{
            .input = TensorView{input, DataType::Float32(), shape, huge_strides},
            .output = MutableTensorView{output, DataType::Float32(), shape, huge_strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSilu, ParamsBuilderRejectsWrongArityAndInvalidViews) {
    const auto kernel = PrepareSiluKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    PreparedKernelParams prepared;
    const std::array<TensorView, 0> no_inputs{};
    const std::array<MutableTensorView, 1> invalid_output{MutableTensorView{}};

    const Status arity_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = no_inputs,
                    .outputs = invalid_output,
                    .attrs = kernel->attrs,
            },
            prepared.storage.data());
    EXPECT_EQ(arity_status.code(), StatusCode::kInvalidArgument) << arity_status.ToString();

    const std::array<TensorView, 1> invalid_input{TensorView{}};
    const Status view_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = invalid_input,
                    .outputs = invalid_output,
                    .attrs = kernel->attrs,
            },
            prepared.storage.data());
    EXPECT_EQ(view_status.code(), StatusCode::kInvalidArgument) << view_status.ToString();
}

} // namespace
