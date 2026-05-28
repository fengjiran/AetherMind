#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/function_operator.h"
#include "aethermind/operators/rms_norm_op.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

/// Reference RMSNorm implementation for validating optimized kernels.
///
/// Computes RMSNorm independently for each row of a [seq_len, hidden] input.
/// This is the numerical oracle — all optimized kernel implementations must
/// produce results indistinguishable from this function.
///
/// \param input    Input data, layout [seq_len][hidden] row-major.
/// \param weight   Weight vector, length hidden_size.
/// \param output   Output buffer, same layout as input.
/// \param seq_len  Number of token rows.
/// \param hidden_size  Hidden dimension size.
/// \param epsilon  Small constant added before rsqrt for numerical stability.
void ReferenceRmsNorm(const float* input,
                      const float* weight,
                      float* output,
                      int64_t seq_len,
                      int64_t hidden_size,
                      float epsilon) noexcept {
    for (int64_t s = 0; s < seq_len; ++s) {
        const float* row_in = input + s * hidden_size;
        float* row_out = output + s * hidden_size;

        double mean_square = 0.0F;
        for (int64_t i = 0; i < hidden_size; ++i) {
            mean_square += row_in[i] * row_in[i];
        }
        mean_square /= static_cast<float>(hidden_size);

        const float inv_rms = 1.0F / std::sqrt(mean_square + epsilon);
        for (int64_t i = 0; i < hidden_size; ++i) {
            row_out[i] = row_in[i] * inv_rms * weight[i];
        }
    }
}

}// namespace

TEST(CpuRmsNormKernel, ComputesExpectedValues) {
    constexpr float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    constexpr int64_t io_shape[2] = {1, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t w_shape[1] = {4};
    constexpr int64_t w_strides[1] = {1};

    const CpuRmsNormParams params{
            .Input = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .Weight = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .Output = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};

    const Status status = CpuRmsNormKernel(KernelInvocation{
                                                   .op_type = OpType::kRmsNorm,
                                                   .selector = {.device_type = DeviceType::kCPU},
                                           },
                                           KernelContext{
                                                   .device = Device::CPU(),
                                                   .packed_params = &params,
                                                   .attrs = std::as_bytes(std::span{&attrs, size_t{1}}),
                                           },
                                           WorkspaceBinding{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.730297, 1e-5);
    EXPECT_NEAR(output[2], 1.095445, 1e-5);
    EXPECT_NEAR(output[3], 1.460593, 1e-5);
}

TEST(CpuRmsNormKernel, CpuBackendResolvedKernelExecutesThroughExecutor) {
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
            .Input = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .Weight = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .Output = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

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

TEST(CpuRmsNormKernel, ExecutionPlanBuilderRunsThroughRmsNormOperator) {
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
            .op_params = RmsNormOp::Params{.epsilon_ = 1.0e-5F},
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
    const CpuRmsNormParams params{
            .Input = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .Weight = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .Output = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };

    ExecutionStep step = plan->steps().front();
    step.packed_params = &params;
    ExecutionPlan executable_plan;
    ASSERT_TRUE(executable_plan.AddStep(step).ok());

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(executable_plan, bindings);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
    EXPECT_NEAR(output[1], 0.365148, 1e-5);
    EXPECT_NEAR(output[2], 1.643168, 1e-5);
    EXPECT_NEAR(output[3], 2.921186, 1e-5);
}

TEST(CpuRmsNormKernel, MultiTokenRmsNorm) {
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
    constexpr int64_t io_shape[2] = {3, 4};
    constexpr int64_t io_strides[2] = {4, 1};
    constexpr int64_t w_shape[1] = {4};
    constexpr int64_t w_strides[1] = {1};

    const CpuRmsNormParams params{
            .Input = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .Weight = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .Output = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};

    const Status status = CpuRmsNormKernel(KernelInvocation{
                                                   .op_type = OpType::kRmsNorm,
                                                   .selector = {.device_type = DeviceType::kCPU},
                                           },
                                           KernelContext{
                                                   .device = Device::CPU(),
                                                   .packed_params = &params,
                                                   .attrs = std::as_bytes(std::span{&attrs, size_t{1}}),
                                           },
                                           WorkspaceBinding{});

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

TEST(CpuRmsNormKernel, MatchesReference) {
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
    constexpr int64_t io_shape[2] = {kSeqLen, kHidden};
    constexpr int64_t io_strides[2] = {kHidden, 1};
    constexpr int64_t w_shape[1] = {kHidden};
    constexpr int64_t w_strides[1] = {1};
    constexpr float kEpsilon = 1.0e-5F;

    const CpuRmsNormParams params{
            .Input = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .Weight = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .Output = MutableTensorView{kernel_output, DataType::Float32(), io_shape, io_strides},
    };
    const CpuRmsNormAttrs attrs{.Epsilon = kEpsilon};

    const Status status = CpuRmsNormKernel(KernelInvocation{
                                                   .op_type = OpType::kRmsNorm,
                                                   .selector = {.device_type = DeviceType::kCPU},
                                           },
                                           KernelContext{
                                                   .device = Device::CPU(),
                                                   .packed_params = &params,
                                                   .attrs = std::as_bytes(std::span{&attrs, size_t{1}}),
                                           },
                                           WorkspaceBinding{});
    ASSERT_TRUE(status.ok()) << status.ToString();

    ReferenceRmsNorm(input, weight, ref_output, kSeqLen, kHidden, kEpsilon);

    for (int i = 0; i < kSeqLen * kHidden; ++i) {
        EXPECT_NEAR(kernel_output[i], ref_output[i], 1e-6)
                << "mismatch at index " << i;
    }
}

}// namespace aethermind
