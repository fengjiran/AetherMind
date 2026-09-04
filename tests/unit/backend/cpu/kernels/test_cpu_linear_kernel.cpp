#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/linear/linear_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

KernelSelector MakeLinearSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareLinearKernel() {
    CpuBackend backend;
    return backend.PrepareKernel(OpType::kLinear, MakeLinearSelector(), OpParams{LinearParams{}});
}

struct LinearTestViews {
    TensorView input_tensor{};
    TensorView weight_tensor{};
    MutableTensorView output_tensor{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildLinearPreparedParams(
        const ResolvedKernel& kernel,
        const LinearTestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 2> inputs{views.input_tensor, views.weight_tensor};
    const std::array<MutableTensorView, 1> outputs{views.output_tensor};
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

Status RunLinearEntry(const ResolvedKernel& kernel,
                      const PreparedKernelParams& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

Status RunLinearEntryWith(const ResolvedKernel& kernel,
                          const LinearTestViews& views) noexcept {
    const auto prepared = BuildLinearPreparedParams(kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return RunLinearEntry(kernel, *prepared);
}

Status RunLinearEntry(const LinearTestViews& views) noexcept {
    const auto kernel = PrepareLinearKernel();
    if (!kernel.ok()) {
        return kernel.status();
    }
    return RunLinearEntryWith(*kernel, views);
}

void ExpectLinearRowsNear(const float* input,
                          const float* weight,
                          const float* output,
                          int64_t row_count,
                          int64_t in_features,
                          int64_t out_features,
                          int64_t input_row_stride,
                          int64_t input_col_stride,
                          int64_t weight_row_stride,
                          int64_t weight_col_stride,
                          int64_t output_row_stride,
                          int64_t output_col_stride) {
    for (int64_t row = 0; row < row_count; ++row) {
        const float* const input_row = input + row * input_row_stride;
        const float* const output_row = output + row * output_row_stride;
        for (int64_t out_feature = 0; out_feature < out_features; ++out_feature) {
            const float* const weight_row = weight + out_feature * weight_row_stride;
            double expected = 0.0;
            for (int64_t in_feature = 0; in_feature < in_features; ++in_feature) {
                expected += static_cast<double>(input_row[in_feature * input_col_stride]) *
                            static_cast<double>(weight_row[in_feature * weight_col_stride]);
            }
            EXPECT_NEAR(output_row[out_feature * output_col_stride],
                        static_cast<float>(expected), 1.0e-6F)
                    << "mismatch at row " << row << ", output feature " << out_feature;
        }
    }
}

TEST(CPUKernelLinear, CpuBackendPreparesPlainF32ReferenceKernel) {
    const auto kernel = PrepareLinearKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kLinear);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::linear_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::LinearF32KernelArgs));
}

TEST(CPUKernelLinear, RejectsSelectorsWithoutPlainF32ReferenceSupport) {
    CpuBackend backend;
    KernelSelector selector = MakeLinearSelector();
    selector.act_dtype = DataType::Float(16);
    const auto fp16 = backend.PrepareKernel(
            OpType::kLinear, selector, OpParams{LinearParams{}});
    EXPECT_FALSE(fp16.ok());
    EXPECT_EQ(fp16.status().code(), StatusCode::kNotFound);

    selector = MakeLinearSelector();
    selector.weight_format = WeightFormat::kPacked;
    const auto packed = backend.PrepareKernel(
            OpType::kLinear, selector, OpParams{LinearParams{}});
    EXPECT_FALSE(packed.ok());
    EXPECT_EQ(packed.status().code(), StatusCode::kNotFound);
}

TEST(CPUKernelLinear, ReferenceExecutesRankOneInput) {
    constexpr int64_t input_shape[1] = {3};
    constexpr int64_t input_strides[1] = {1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[1] = {2};
    constexpr int64_t output_strides[1] = {1};
    constexpr float input[3] = {1.0F, -2.0F, 0.5F};
    constexpr float weight[6] = {2.0F, 1.0F, -1.0F, -0.5F, 3.0F, 4.0F};
    float output[2] = {};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input, weight, output, 1, 3, 2, 0, 1, 3, 1, 0, 1);
}

TEST(CPUKernelLinear, ReferenceExecutesRankTwoInput) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F};
    constexpr float weight[6] = {1.0F, -2.0F, 0.5F, 3.0F, 1.0F, -1.0F};
    float output[4] = {};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input, weight, output, 2, 3, 2, 3, 1, 3, 1, 2, 1);
}

TEST(CPUKernelLinear, ReferenceExecutesCollapsibleRankFourPaddedRows) {
    constexpr int64_t input_shape[4] = {2, 2, 2, 3};
    constexpr int64_t input_strides[4] = {20, 10, 5, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[4] = {2, 2, 2, 2};
    constexpr int64_t output_strides[4] = {28, 14, 7, 1};
    std::array<float, 38> input{};
    constexpr float weight[6] = {1.0F, -0.5F, 2.0F, -1.0F, 3.0F, 0.25F};
    std::array<float, 51> output{};
    for (int64_t row = 0; row < 8; ++row) {
        for (int64_t feature = 0; feature < 3; ++feature) {
            input[static_cast<size_t>(row * 5 + feature)] =
                    static_cast<float>(row * 3 + feature - 7) * 0.25F;
        }
    }

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input.data(), weight, output.data(), 8, 3, 2, 5, 1, 3, 1, 7, 1);
}

TEST(CPUKernelLinear, ReferenceExecutesPositiveColumnStrides) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {7, 2};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {8, 2};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {8, 3};
    std::array<float, 12> input{};
    std::array<float, 13> weight{};
    std::array<float, 12> output{};
    input[0] = 1.0F;
    input[2] = -2.0F;
    input[4] = 0.5F;
    input[7] = 3.0F;
    input[9] = 4.0F;
    input[11] = -1.0F;
    weight[0] = 2.0F;
    weight[2] = -1.0F;
    weight[4] = 0.5F;
    weight[8] = -3.0F;
    weight[10] = 1.0F;
    weight[12] = 2.0F;

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input.data(), weight.data(), output.data(), 2, 3, 2, 7, 2, 8, 2, 8, 3);
}

TEST(CPUKernelLinear, ZeroLeadingDimensionIsSuccessfulNoOp) {
    constexpr int64_t input_shape[3] = {2, 0, 3};
    constexpr int64_t input_strides[3] = {0, 3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[3] = {2, 0, 2};
    constexpr int64_t output_strides[3] = {0, 2, 1};
    constexpr float weight[6] = {};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{nullptr, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{nullptr, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelLinear, ZeroOutputFeatureDimensionIsSuccessfulNoOp) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {0, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 0};
    constexpr int64_t output_strides[2] = {1, 1};
    constexpr float input[6] = {};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{nullptr, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{nullptr, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelLinear, ZeroInputFeatureDimensionWritesZero) {
    constexpr int64_t input_shape[2] = {2, 0};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t weight_shape[2] = {3, 0};
    constexpr int64_t weight_strides[2] = {1, 1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 1};
    float output[6] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{nullptr, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{nullptr, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    for (float value: output) {
        EXPECT_EQ(value, 0.0F);
    }
}

TEST(CPUKernelLinearEntry, RejectsInvalidDtypeRankAndShape) {
    constexpr int64_t input_shape[2] = {1, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t weight_shape[2] = {2, 2};
    constexpr int64_t weight_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {1, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float data[4] = {};
    float output[2] = {};

    const Status wrong_dtype = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Double(), input_shape, input_strides},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(wrong_dtype.code(), StatusCode::kInvalidArgument) << wrong_dtype.ToString();

    const Status rank_zero = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), {}, {}},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), {}, {}},
    });
    EXPECT_EQ(rank_zero.code(), StatusCode::kInvalidArgument) << rank_zero.ToString();

    constexpr int64_t bad_weight_shape[2] = {2, 3};
    const Status mismatched_shape = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{data, DataType::Float32(), bad_weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(mismatched_shape.code(), StatusCode::kInvalidArgument) << mismatched_shape.ToString();
}

TEST(CPUKernelLinearEntry, RejectsInvalidViewAndNonPositiveStrides) {
    constexpr int64_t input_shape[2] = {1, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t zero_strides[2] = {0, 0};
    constexpr int64_t weight_shape[2] = {2, 2};
    constexpr int64_t weight_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {1, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float data[4] = {};
    float output[2] = {};

    const Status invalid_view = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(invalid_view.code(), StatusCode::kInvalidArgument) << invalid_view.ToString();

    const Status bad_stride = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), input_shape, zero_strides},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(bad_stride.code(), StatusCode::kInvalidArgument) << bad_stride.ToString();
}

TEST(CPUKernelLinearEntry, RejectsRowCountAndAddressOffsetOverflow) {
    constexpr int64_t overflow_shape[3] = {std::numeric_limits<int64_t>::max(), 2, 1};
    constexpr int64_t overflow_strides[3] = {2, 1, 1};
    constexpr int64_t weight_shape[2] = {1, 1};
    constexpr int64_t weight_strides[2] = {1, 1};
    float scalar = 1.0F;

    const Status row_count_overflow = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{&scalar, DataType::Float32(), overflow_shape, overflow_strides},
            .weight_tensor = TensorView{&scalar, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{&scalar, DataType::Float32(), overflow_shape, overflow_strides},
    });
    EXPECT_EQ(row_count_overflow.code(), StatusCode::kInvalidArgument)
            << row_count_overflow.ToString();

    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    constexpr int64_t output_shape[2] = {2, 1};
    constexpr int64_t output_strides[2] = {1, 1};
    float output[2] = {};
    const Status address_overflow = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{&scalar, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{&scalar, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(address_overflow.code(), StatusCode::kInvalidArgument) << address_overflow.ToString();
}

TEST(CPUKernelLinearEntry, RejectsNonCollapsibleOrOverlappingOutputRows) {
    constexpr int64_t high_rank_input_shape[3] = {2, 2, 2};
    constexpr int64_t noncollapsible_input_strides[3] = {5, 2, 1};
    constexpr int64_t high_rank_output_shape[3] = {2, 2, 1};
    constexpr int64_t output_strides[3] = {2, 1, 1};
    constexpr int64_t weight_shape[2] = {1, 2};
    constexpr int64_t weight_strides[2] = {2, 1};
    std::array<float, 8> input{};
    constexpr float weight[2] = {};
    std::array<float, 4> output{};

    const Status noncollapsible = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), high_rank_input_shape, noncollapsible_input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), high_rank_output_shape, output_strides},
    });
    EXPECT_EQ(noncollapsible.code(), StatusCode::kUnimplemented) << noncollapsible.ToString();

    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t overlap_shape[2] = {2, 2};
    constexpr int64_t overlap_strides[2] = {1, 1};
    const Status overlap = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), overlap_shape, overlap_strides},
    });
    EXPECT_EQ(overlap.code(), StatusCode::kInvalidArgument) << overlap.ToString();
}

TEST(CPUKernelLinearEntry, RejectsOutputBasePointerAliases) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    std::array<float, 6> input{};
    std::array<float, 6> weight{};

    const Status aliases_input = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{input.data(), DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(aliases_input.code(), StatusCode::kInvalidArgument) << aliases_input.ToString();

    const Status aliases_weight = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{weight.data(), DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(aliases_weight.code(), StatusCode::kInvalidArgument) << aliases_weight.ToString();
}

TEST(CPUKernelLinear, ExecutionPlanBuilderRunsPreparedReferenceKernel) {
    RuntimeBuilder builder;
    Runtime runtime = builder.Build();

    const SymbolicShape activation_shape = StaticShape({2, 3});
    const SymbolicShape weight_shape = StaticShape({2, 3});
    const std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = activation_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    const auto analyzed = InferOperator(OpType::kLinear, OpParams{LinearParams{}}, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kLinear,
                    .selector = MakeLinearSelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = OpParams{LinearParams{}},
            },
    };
    const auto plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_raw_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 4.0F};
    constexpr float weight[6] = {1.0F, -2.0F, 0.5F, 3.0F, 1.0F, -1.0F};
    float output[4] = {};

    test::ExecutionBindingCollector collector(*plan, runtime.GetAllocator(Device::CPU()));
    collector.Set(0, StepTensorBinding{
                             .inputs = {
                                     TensorView{input, DataType::Float32(), input_shape, input_strides},
                                     TensorView{weight, DataType::Float32(), weight_raw_shape, weight_strides},
                             },
                             .outputs = {
                                     MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
                             },
                     });
    auto context = collector.CreateContext();
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const Status status = Executor::Execute(*plan, *context);
    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input, weight, output, 2, 3, 2, 3, 1, 3, 1, 2, 1);
}

} // namespace
