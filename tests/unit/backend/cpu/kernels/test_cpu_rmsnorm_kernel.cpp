#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <gtest/gtest.h>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

constexpr float kEpsilon = 1.0e-5F;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

KernelSelector MakeRmsNormSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<CpuFeaturePolicy> MakeScalarCpuFeaturePolicy() {
    const auto capabilities = cpu::DetectCpuCapabilities();
    if (!capabilities.ok()) {
        return capabilities.status();
    }
    return CpuFeaturePolicy{
            .disabled_features = capabilities->usable_features,
    };
}

StatusOr<ResolvedKernel> PrepareRmsNormKernel(float epsilon = kEpsilon) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kRmsNorm,
            MakeRmsNormSelector(),
            OpParams{RmsNormParams{.eps = epsilon}});
}

StatusOr<ResolvedKernel> PrepareScalarRmsNormKernel(float epsilon = kEpsilon) {
    const auto policy = MakeScalarCpuFeaturePolicy();
    if (!policy.ok()) {
        return policy.status();
    }
    CpuBackend backend(*policy);
    return backend.PrepareKernel(
            OpType::kRmsNorm,
            MakeRmsNormSelector(),
            OpParams{RmsNormParams{.eps = epsilon}});
}

Status RunRmsNormEntry(const ResolvedKernel& kernel,
                       const cpu::detail::RmsNormKernelParams& params) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = &params,
            .attrs = kernel.attrs,
    });
}

Status RunScalarRmsNormEntry(const cpu::detail::RmsNormKernelParams& params) noexcept {
    const StatusOr<ResolvedKernel> kernel = PrepareScalarRmsNormKernel();
    if (!kernel.ok()) {
        return kernel.status();
    }
    return RunRmsNormEntry(*kernel, params);
}

bool IsAvx2FmaRmsNormKernel(const ResolvedKernel& kernel) noexcept {
    return kernel.debug_name != nullptr &&
           std::string_view{kernel.debug_name} == std::string_view{"cpu::rmsnorm_f32_avx2_fma"};
}

void ExpectRowsNear(const float* input,
                    const float* weight,
                    const float* output,
                    int64_t row_count,
                    int64_t hidden_size,
                    int64_t input_row_stride,
                    int64_t input_col_stride,
                    int64_t weight_stride,
                    int64_t output_row_stride,
                    int64_t output_col_stride,
                    float tolerance = 1.0e-6F) {
    for (int64_t row = 0; row < row_count; ++row) {
        const float* const input_row = input + row * input_row_stride;
        const float* const output_row = output + row * output_row_stride;

        double sum_sq = 0.0;
        for (int64_t col = 0; col < hidden_size; ++col) {
            const double value = input_row[col * input_col_stride];
            sum_sq += value * value;
        }

        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(hidden_size) +
                                               static_cast<double>(kEpsilon));
        for (int64_t col = 0; col < hidden_size; ++col) {
            const auto expected = static_cast<float>(
                    static_cast<double>(input_row[col * input_col_stride]) * inv_rms *
                    static_cast<double>(weight[col * weight_stride]));
            EXPECT_NEAR(output_row[col * output_col_stride], expected, tolerance)
                    << "mismatch at row " << row << ", col " << col;
        }
    }
}

TEST(CPUKernelRmsNorm, CpuBackendPreparedKernelExecutesRankTwoInput) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input, weight, output, 1, 4, 4, 1, 1, 4, 1);
}

TEST(CPUKernelRmsNorm, ScalarSupportsRankOneInput) {
    constexpr float input[4] = {1.0F, -2.0F, 3.0F, -4.0F};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {};
    constexpr int64_t io_shape[1] = {4};
    constexpr int64_t io_strides[1] = {1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input, weight, output, 1, 4, 0, 1, 1, 0, 1);
}

TEST(CPUKernelRmsNorm, ScalarSupportsCollapsibleRankThreeInput) {
    constexpr int64_t io_shape[3] = {2, 3, 4};
    constexpr int64_t io_strides[3] = {12, 4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    std::array<float, 24> input{};
    std::array<float, 24> output{};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    for (int64_t index = 0; index < 24; ++index) {
        input[static_cast<size_t>(index)] = static_cast<float>(index - 9) * 0.25F;
    }

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input.data(), DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), io_shape, io_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input.data(), weight, output.data(), 6, 4, 4, 1, 1, 4, 1);
}

TEST(CPUKernelRmsNorm, ScalarSupportsCollapsibleRankFourPaddedRows) {
    constexpr int64_t io_shape[4] = {2, 2, 3, 4};
    constexpr int64_t io_strides[4] = {30, 15, 5, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    std::array<float, 59> input{};
    std::array<float, 59> output{};
    constexpr float weight[4] = {1.0F, 1.25F, 0.75F, 0.5F};
    for (int64_t row = 0; row < 12; ++row) {
        for (int64_t col = 0; col < 4; ++col) {
            input[static_cast<size_t>(row * 5 + col)] =
                    static_cast<float>(row * 4 + col - 11) * 0.125F;
        }
    }

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input.data(), DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), io_shape, io_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input.data(), weight, output.data(), 12, 4, 5, 1, 1, 5, 1);
}

TEST(CPUKernelRmsNorm, ScalarSupportsExactInPlace) {
    constexpr int64_t io_shape[3] = {2, 2, 4};
    constexpr int64_t io_strides[3] = {8, 4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    std::array<float, 16> storage{};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    for (int64_t index = 0; index < 16; ++index) {
        storage[static_cast<size_t>(index)] = static_cast<float>(index - 7) * 0.5F;
    }
    const std::array<float, 16> original = storage;

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{storage.data(), DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{storage.data(), DataType::Float32(), io_shape, io_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(original.data(), weight, storage.data(), 4, 4, 4, 1, 1, 4, 1);
}

TEST(CPUKernelRmsNorm, ScalarSupportsPositiveInnerStrides) {
    constexpr int64_t io_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {7, 2};
    constexpr int64_t output_strides[2] = {8, 2};
    constexpr int64_t weight_shape[1] = {3};
    constexpr int64_t weight_strides[1] = {2};
    constexpr float input[12] = {
            1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 0.0F, -1.0F, 0.0F, 0.5F, 0.0F, 4.0F};
    constexpr float weight[5] = {1.0F, 0.0F, 0.5F, 0.0F, 1.5F};
    std::array<float, 14> output{};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input, DataType::Float32(), io_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), io_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input, weight, output.data(), 2, 3, 7, 2, 2, 8, 2);
}

TEST(CPUKernelRmsNorm, Avx2FmaPreparedKernelMatchesDoubleReferenceWithTail) {
    const StatusOr<ResolvedKernel> kernel = PrepareRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    if (!IsAvx2FmaRmsNormKernel(*kernel)) {
        GTEST_SKIP() << "AVX2+FMA RMSNorm kernel is unavailable";
    }

    constexpr int64_t row_count = 2;
    constexpr int64_t hidden_size = 33;
    constexpr int64_t io_shape[2] = {row_count, hidden_size};
    constexpr int64_t io_strides[2] = {hidden_size, 1};
    constexpr int64_t weight_shape[1] = {hidden_size};
    constexpr int64_t weight_strides[1] = {1};
    std::array<float, row_count * hidden_size> input{};
    std::array<float, hidden_size> weight{};
    std::array<float, row_count * hidden_size> output{};
    for (int64_t index = 0; index < row_count * hidden_size; ++index) {
        input[static_cast<size_t>(index)] = static_cast<float>(index - 31) * 0.03125F;
    }
    for (int64_t index = 0; index < hidden_size; ++index) {
        weight[static_cast<size_t>(index)] = 0.75F + static_cast<float>(index % 7) * 0.125F;
    }

    const Status status = RunRmsNormEntry(*kernel, cpu::detail::RmsNormKernelParams{
                                                           .input_tensor = TensorView{input.data(), DataType::Float32(), io_shape, io_strides},
                                                           .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
                                                           .output_tensor = MutableTensorView{output.data(), DataType::Float32(), io_shape, io_strides},
                                                   });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input.data(), weight.data(), output.data(), row_count, hidden_size, hidden_size, 1, 1, hidden_size, 1, 1.0e-5F);
}

TEST(CPUKernelRmsNormEntry, ZeroLeadingDimensionIsSuccessfulNoOp) {
    constexpr int64_t io_shape[3] = {2, 0, 4};
    constexpr int64_t io_strides[3] = {0, 4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{nullptr, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{nullptr, DataType::Float32(), io_shape, io_strides},
    });

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsNonCollapsibleLeadingDimensions) {
    constexpr int64_t io_shape[3] = {2, 3, 4};
    constexpr int64_t input_strides[3] = {13, 4, 1};
    constexpr int64_t output_strides[3] = {12, 4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    std::array<float, 25> input{};
    std::array<float, 24> output{};
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input.data(), DataType::Float32(), io_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), io_shape, output_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsRowCountOverflow) {
    constexpr int64_t io_shape[3] = {std::numeric_limits<int64_t>::max(), 2, 1};
    constexpr int64_t io_strides[3] = {2, 1, 1};
    constexpr int64_t weight_shape[1] = {1};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float input = 1.0F;
    constexpr float weight = 1.0F;
    float output = 0.0F;

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{&input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{&weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{&output, DataType::Float32(), io_shape, io_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsAddressOffsetOverflow) {
    constexpr int64_t io_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr int64_t weight_shape[1] = {2};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float input = 1.0F;
    constexpr float weight[2] = {1.0F, 1.0F};
    float output[4] = {};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{&input, DataType::Float32(), io_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, output_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsRankZeroInput) {
    constexpr float input = 1.0F;
    constexpr float weight = 1.0F;
    float output = 0.0F;
    constexpr int64_t weight_shape[1] = {1};
    constexpr int64_t weight_strides[1] = {1};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{&input, DataType::Float32(), {}, {}},
            .weight_tensor = TensorView{&weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{&output, DataType::Float32(), {}, {}},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsZeroHiddenSize) {
    constexpr int64_t io_shape[2] = {1, 0};
    constexpr int64_t io_strides[2] = {1, 1};
    constexpr int64_t weight_shape[1] = {0};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float input = 0.0F;
    constexpr float weight = 1.0F;
    float output = 0.0F;

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{&input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{&weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{&output, DataType::Float32(), io_shape, io_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsMismatchedOutputShape) {
    constexpr int64_t input_shape[2] = {1, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    constexpr int64_t output_shape[2] = {2, 4};
    constexpr int64_t output_strides[2] = {4, 1};
    constexpr int64_t weight_shape[1] = {4};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[8] = {};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsNonFloat32Tensors) {
    constexpr int64_t io_shape[2] = {1, 2};
    constexpr int64_t io_strides[2] = {2, 1};
    constexpr int64_t weight_shape[1] = {2};
    constexpr int64_t weight_strides[1] = {1};
    constexpr float input[2] = {1.0F, 2.0F};
    constexpr float weight[2] = {1.0F, 1.0F};
    float output[2] = {};

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{input, DataType::Double(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, RejectsNonPositiveStridesForNonEmptyTensor) {
    constexpr int64_t io_shape[2] = {1, 1};
    constexpr int64_t io_strides[2] = {0, 0};
    constexpr int64_t weight_shape[1] = {1};
    constexpr int64_t weight_strides[1] = {0};
    constexpr float input = 1.0F;
    constexpr float weight = 1.0F;
    float output = 0.0F;

    const Status status = RunScalarRmsNormEntry(cpu::detail::RmsNormKernelParams{
            .input_tensor = TensorView{&input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{&weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{&output, DataType::Float32(), io_shape, io_strides},
    });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNormEntry, PrepareRejectsInvalidEpsilon) {
    const StatusOr<ResolvedKernel> kernel = PrepareScalarRmsNormKernel(0.0F);
    ASSERT_FALSE(kernel.ok());
    EXPECT_EQ(kernel.status().code(), StatusCode::kInvalidArgument) << kernel.status().ToString();
}

TEST(CPUKernelRmsNormEntry, Avx2RejectsNonUnitColumnStridesWhenAvailable) {
    const StatusOr<ResolvedKernel> kernel = PrepareRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    if (!IsAvx2FmaRmsNormKernel(*kernel)) {
        GTEST_SKIP() << "AVX2+FMA RMSNorm kernel was not compiled";
    }

    constexpr int64_t io_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {7, 2};
    constexpr int64_t output_strides[2] = {8, 2};
    constexpr int64_t weight_shape[1] = {3};
    constexpr int64_t weight_strides[1] = {2};
    constexpr float input[12] = {
            1.0F, 0.0F, 2.0F, 0.0F, 3.0F, 0.0F, 0.0F, -1.0F, 0.0F, 0.5F, 0.0F, 4.0F};
    constexpr float weight[5] = {1.0F, 0.0F, 0.5F, 0.0F, 1.5F};
    float output[14] = {};

    const Status status = RunRmsNormEntry(*kernel, cpu::detail::RmsNormKernelParams{
                                                           .input_tensor = TensorView{input, DataType::Float32(), io_shape, input_strides},
                                                           .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
                                                           .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, output_strides},
                                                   });

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelRmsNorm, ExecutionPlanBuilderRunsResolvedKernel) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    const SymbolicShape activation_shape = StaticShape({1, 4});
    const SymbolicShape weight_shape = StaticShape({4});
    std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = activation_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    const auto analyzed = InferOperator(OpType::kRmsNorm,
                                        OpParams{RmsNormParams{.eps = kEpsilon}},
                                        inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kRmsNorm,
                    .selector = MakeRmsNormSelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = OpParams{RmsNormParams{.eps = kEpsilon}},
            },
    };
    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t raw_weight_shape[1] = {4};
    constexpr int64_t raw_weight_strides[1] = {1};

    RuntimeBindingContext bindings;
    test::ExecutionBindingCollector collector(*plan, runtime.GetAllocator(Device::CPU()));
    collector.Set(0, StepTensorBinding{
                             .inputs = {
                                     TensorView{input, DataType::Float32(), io_shape, io_strides},
                                     TensorView{weight, DataType::Float32(), raw_weight_shape, raw_weight_strides},
                             },
                             .outputs = {
                                     MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
                             },
                     });
    ASSERT_TRUE(collector.Install(bindings).ok());

    const Status status = Executor::Execute(*plan, bindings);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectRowsNear(input, weight, output, 1, 4, 4, 1, 1, 4, 1);
}

} // namespace
