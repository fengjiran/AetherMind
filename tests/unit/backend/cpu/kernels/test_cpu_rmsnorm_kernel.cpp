#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/kernels/rmsnorm/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/function_operator.h"
#include "aethermind/operators/rmsnorm_op.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace aethermind {
namespace {

Status RunRmsNormEntry(const CpuRmsNormParams& params, float epsilon) noexcept {
    return CpuRmsNormKernelEntry_FP32_AVX2(KernelContext{
            .packed_params = &params,
            .attrs = std::as_bytes(std::span{&epsilon, size_t{1}}),
    });
}

Status RunRmsNormEntry(const CpuRmsNormParams& params) noexcept {
    return RunRmsNormEntry(params, 1.0e-5F);
}

void ExpectInvalidRmsNormEntry(const CpuRmsNormParams& params) {
    const Status status = RunRmsNormEntry(params);
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

}// namespace

TEST(CPUKernelRmsNorm, ComputesExpectedValues) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const Status status = LaunchRmsNorm(RmsNormArgs{
            .input = input,
            .weight = weight,
            .output = output,
            .seq_len = 1,
            .hidden_size = 4,
            .input_row_stride = 4,
            .input_col_stride = 1,
            .weight_stride = 1,
            .output_row_stride = 4,
            .output_col_stride = 1,
            .epsilon = 1.0e-5F,
            .dtype = DataType::Float32(),
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.730297, 1e-5);
    EXPECT_NEAR(output[2], 1.095445, 1e-5);
    EXPECT_NEAR(output[3], 1.460593, 1e-5);
}

TEST(CPUKernelRmsNorm, CpuBackendResolvedKernelExecutesThroughExecutor) {
    CpuBackend backend;
    const StatusOr<ResolvedKernel> resolved = backend.ResolveKernelInfo(
            OpType::kRmsNorm,
            KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .activation_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            });
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();

    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t w_shape[1] = {4};
    constexpr int64_t w_strides[1] = {1};
    const CpuRmsNormParams params{
            .input_tensor = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const float epsilon = 1.0e-5F;
    const auto attrs_bytes = std::as_bytes(std::span{&epsilon, size_t{1}});

    ExecutionPlan plan;
    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .selector = {
                                             .device_type = DeviceType::kCPU,
                                             .activation_dtype = DataType::Float32(),
                                             .weight_dtype = DataType::Float32(),
                                             .weight_format = WeightFormat::kPlain,
                                             .isa = IsaLevel::kScalar,
                                             .phase = ExecPhase::kBoth,
                                     },
                                     .op = std::make_shared<FunctionOperator>(resolved->op_type, resolved->fn, attrs_bytes, resolved->debug_name),
                                     .packed_params = &params,
                                     .workspace_requirement = {},
                                     .debug_name = resolved->debug_name,
                             })
                        .ok());

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.365148, 1e-5);
    EXPECT_NEAR(output[2], 1.643168, 1e-5);
    EXPECT_NEAR(output[3], 2.921186, 1e-5);
}

TEST(CPUKernelRmsNorm, ExecutionPlanBuilderRunsThroughRmsNormOperator) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .op_params = RmsNormOp::Params{.epsilon = 1.0e-5F},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().op, nullptr);

    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t w_shape[1] = {4};
    constexpr int64_t w_strides[1] = {1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView{input, DataType::Float32(), io_shape, io_strides},
                                                     TensorView{weight, DataType::Float32(), w_shape, w_strides},
                                             },
                                             .outputs = {
                                                     MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
                                             },
                                     });

    const Status status = Executor::Execute(*plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.365148, 1e-5);
    EXPECT_NEAR(output[2], 1.643168, 1e-5);
    EXPECT_NEAR(output[3], 2.921186, 1e-5);
}

TEST(CPUKernelRmsNorm, ExecutionPlanBuilderRmsNormFailsWithoutTensorBinding) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRmsNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .op_params = RmsNormOp::Params{.epsilon = 1.0e-5F},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(*plan, bindings);

    EXPECT_FALSE(status.ok());
}

TEST(CPUKernelRmsNorm, MultiTokenRmsNorm) {
    // [seq_len=3, hidden=4] — three rows, each independently normalized.
    constexpr float input[12] = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            0.5F,
            1.0F,
            1.5F,
            2.0F,
            4.0F,
            3.0F,
            2.0F,
            1.0F,
    };
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[12] = {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    const Status status = LaunchRmsNorm(RmsNormArgs{
            .input = input,
            .weight = weight,
            .output = output,
            .seq_len = 3,
            .hidden_size = 4,
            .input_row_stride = 4,
            .input_col_stride = 1,
            .weight_stride = 1,
            .output_row_stride = 4,
            .output_col_stride = 1,
            .epsilon = 1.0e-5F,
            .dtype = DataType::Float32(),
    });

    ASSERT_TRUE(status.ok()) << status.ToString();

    // Row 0: {1, 2, 3, 4} → same as single-token case
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.730297, 1e-5);
    EXPECT_NEAR(output[2], 1.095445, 1e-5);
    EXPECT_NEAR(output[3], 1.460593, 1e-5);

    // Row 1: {0.5, 1.0, 1.5, 2.0} → mean_square = (0.25+1+2.25+4)/4 = 1.875
    // rms = sqrt(1.875+1e-5) ≈ 1.369306, inv_rms ≈ 0.730297
    // output = input * inv_rms
    EXPECT_NEAR(output[4], 0.365148, 1e-5);
    EXPECT_NEAR(output[5], 0.730297, 1e-5);
    EXPECT_NEAR(output[6], 1.095445, 1e-5);
    EXPECT_NEAR(output[7], 1.460593, 1e-5);

    // Row 2: {4, 3, 2, 1} → same values as row 0 but reversed; rms is same
    EXPECT_NEAR(output[8], 1.460593, 1e-5);
    EXPECT_NEAR(output[9], 1.095445, 1e-5);
    EXPECT_NEAR(output[10], 0.730297, 1e-5);
    EXPECT_NEAR(output[11], 0.365148, 1e-5);
}

TEST(CPUKernelRmsNorm, MatchesReference) {
    // Setup: 3 tokens, hidden=4, non-uniform values + non-uniform weight.
    constexpr int64_t kSeqLen = 3;
    constexpr int64_t kHidden = 4;
    constexpr float input[12] = {
            1.0F,
            2.0F,
            3.0F,
            4.0F,
            0.5F,
            1.0F,
            1.5F,
            2.0F,
            4.0F,
            3.0F,
            2.0F,
            1.0F,
    };
    constexpr float weight[4] = {1.0F, 0.5F, 1.5F, 2.0F};
    float kernel_output[12] = {};
    float ref_output[12] = {};
    constexpr float kEpsilon = 1.0e-5F;

    const Status status1 = RmsNormKernel_CPU_FP32_AVX2(
            RmsNormFp32KernelArgs{
                    .input = input,
                    .weight = weight,
                    .output = kernel_output,
                    .seq_len = kSeqLen,
                    .hidden_size = kHidden,
                    .input_row_stride = kHidden,
                    .input_col_stride = 1,
                    .weight_stride = 1,
                    .output_row_stride = kHidden,
                    .output_col_stride = 1,
                    .epsilon = kEpsilon});
    ASSERT_TRUE(status1.ok()) << status1.ToString();

    const Status status2 = RmsNormKernel_CPU_FP32_Scalar(
            RmsNormFp32KernelArgs{
                    .input = input,
                    .weight = weight,
                    .output = ref_output,
                    .seq_len = kSeqLen,
                    .hidden_size = kHidden,
                    .input_row_stride = kHidden,
                    .input_col_stride = 1,
                    .weight_stride = 1,
                    .output_row_stride = kHidden,
                    .output_col_stride = 1,
                    .epsilon = kEpsilon});
    ASSERT_TRUE(status2.ok()) << status2.ToString();

    for (int i = 0; i < kSeqLen * kHidden; ++i) {
        EXPECT_NEAR(kernel_output[i], ref_output[i], 1e-6) << "mismatch at index " << i;
    }
}

TEST(CPUKernelRmsNorm, StridedTypedArgsMatchesReference) {
    constexpr int64_t kSeqLen = 2;
    constexpr int64_t kHidden = 3;
    constexpr int64_t kInputRowStride = 7;
    constexpr int64_t kInputColStride = 2;
    constexpr int64_t kWeightStride = 2;
    constexpr int64_t kOutputRowStride = 8;
    constexpr int64_t kOutputColStride = 2;
    constexpr float kEpsilon = 1.0e-5F;

    std::array<float, 12> input{};
    std::array<float, 5> weight{};
    std::array<float, 14> kernel_output{};
    std::array<float, 14> ref_output{};

    input[0] = 1.0F;
    input[2] = 2.0F;
    input[4] = 3.0F;
    input[7] = -1.0F;
    input[9] = 0.5F;
    input[11] = 4.0F;
    weight[0] = 1.0F;
    weight[2] = 0.5F;
    weight[4] = 1.5F;

    for (int64_t row = 0; row < kSeqLen; ++row) {
        const float* const row_in = input.data() + row * kInputRowStride;
        float* const row_out = kernel_output.data() + row * kOutputRowStride;

        double sum_sq = 0.0;
        for (int64_t j = 0; j < kHidden; ++j) {
            const auto x = static_cast<double>(row_in[j * kInputColStride]);
            sum_sq += x * x;
        }

        const double mean_sq = sum_sq / static_cast<double>(kHidden);
        const double inv_rms = 1.0 / std::sqrt(mean_sq + static_cast<double>(kEpsilon));
        for (int64_t j = 0; j < kHidden; ++j) {
            row_out[j * kOutputColStride] = static_cast<float>(
                    static_cast<double>(row_in[j * kInputColStride]) * inv_rms *
                    static_cast<double>(weight[j * kWeightStride]));
        }
    }

    const Status status = RmsNormKernel_CPU_FP32_Scalar(RmsNormFp32KernelArgs{
            .input = input.data(),
            .weight = weight.data(),
            .output = ref_output.data(),
            .seq_len = kSeqLen,
            .hidden_size = kHidden,
            .input_row_stride = kInputRowStride,
            .input_col_stride = kInputColStride,
            .weight_stride = kWeightStride,
            .output_row_stride = kOutputRowStride,
            .output_col_stride = kOutputColStride,
            .epsilon = kEpsilon,
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (int64_t row = 0; row < kSeqLen; ++row) {
        for (int64_t col = 0; col < kHidden; ++col) {
            const int64_t offset = row * kOutputRowStride + col * kOutputColStride;
            EXPECT_NEAR(kernel_output[static_cast<size_t>(offset)], ref_output[static_cast<size_t>(offset)], 1e-6)
                    << "mismatch at row " << row << ", col " << col;
        }
    }
}

TEST(CPUKernelRmsNormEntry, RejectsNullDataPointers) {
    constexpr float kInput[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float kWeight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};
    constexpr int64_t kIoShape[2] = {1, 4};
    constexpr int64_t kIoStrides[2] = {4, 1};
    constexpr int64_t kWeightShape[1] = {4};
    constexpr int64_t kWeightStrides[1] = {1};

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{nullptr, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{nullptr, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{nullptr, DataType::Float32(), kIoShape, kIoStrides},
    });
}

TEST(CPUKernelRmsNormEntry, RejectsNonFloat32Dtypes) {
    constexpr float kInput[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float kWeight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};
    constexpr int64_t kIoShape[2] = {1, 4};
    constexpr int64_t kIoStrides[2] = {4, 1};
    constexpr int64_t kWeightShape[1] = {4};
    constexpr int64_t kWeightStrides[1] = {1};

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Double(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Double(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Double(), kIoShape, kIoStrides},
    });
}

TEST(CPUKernelRmsNormEntry, RejectsZeroHiddenSize) {
    constexpr float kInput[1] = {0.0F};
    constexpr float kWeight[1] = {1.0F};
    float output[1] = {};
    constexpr int64_t kIoShape[2] = {1, 0};
    constexpr int64_t kIoStrides[2] = {1, 1};
    constexpr int64_t kWeightShape[1] = {0};
    constexpr int64_t kWeightStrides[1] = {1};

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });
}

TEST(CPUKernelRmsNormEntry, RejectsZeroStrides) {
    constexpr float kInput[1] = {1.0F};
    constexpr float kWeight[1] = {1.0F};
    float output[1] = {};
    constexpr int64_t kIoShape[2] = {1, 1};
    constexpr int64_t kIoStrides[2] = {0, 0};
    constexpr int64_t kWeightShape[1] = {1};
    constexpr int64_t kWeightStrides[1] = {0};

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    });
}

TEST(CPUKernelRmsNormEntry, RejectsMismatchedOutputShape) {
    constexpr float kInput[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float kWeight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[8] = {};
    constexpr int64_t kInputShape[2] = {1, 4};
    constexpr int64_t kInputStrides[2] = {4, 1};
    constexpr int64_t kOutputShape[2] = {2, 4};
    constexpr int64_t kOutputStrides[2] = {4, 1};
    constexpr int64_t kWeightShape[1] = {4};
    constexpr int64_t kWeightStrides[1] = {1};

    ExpectInvalidRmsNormEntry(CpuRmsNormParams{
            .input_tensor = TensorView{kInput, DataType::Float32(), kInputShape, kInputStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kOutputShape, kOutputStrides},
    });
}

TEST(CPUKernelRmsNormEntry, RejectsInvalidEpsilon) {
    constexpr float kInput[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float kWeight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {};
    constexpr int64_t kIoShape[2] = {1, 4};
    constexpr int64_t kIoStrides[2] = {4, 1};
    constexpr int64_t kWeightShape[1] = {4};
    constexpr int64_t kWeightStrides[1] = {1};
    const CpuRmsNormParams params{
            .input_tensor = TensorView{kInput, DataType::Float32(), kIoShape, kIoStrides},
            .weight_tensor = TensorView{kWeight, DataType::Float32(), kWeightShape, kWeightStrides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), kIoShape, kIoStrides},
    };
    const float epsilon = 0.0F;

    const Status status = RunRmsNormEntry(params, epsilon);
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

}// namespace aethermind
