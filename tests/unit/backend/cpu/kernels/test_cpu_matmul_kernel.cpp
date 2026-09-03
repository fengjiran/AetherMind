#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/matmul/matmul_internal.h"
#include "execution/test_execution_binding_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

KernelSelector MakeMatMulSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareMatMulKernel(bool transpose_rhs = false) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kMatMul, MakeMatMulSelector(), OpParams{MatMulParams{.transpose_rhs = transpose_rhs}});
}

struct MatMulTestViews {
    TensorView lhs{};
    TensorView rhs{};
    MutableTensorView output{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildMatMulPreparedParams(
        const ResolvedKernel& kernel,
        const MatMulTestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 2> inputs{views.lhs, views.rhs};
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

Status RunMatMulEntry(const ResolvedKernel& kernel,
                      const PreparedKernelParams& prepared) noexcept {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

Status RunMatMulEntryWith(const ResolvedKernel& kernel,
                          const MatMulTestViews& views) noexcept {
    const auto prepared = BuildMatMulPreparedParams(kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return RunMatMulEntry(kernel, *prepared);
}

Status RunMatMulEntry(const MatMulTestViews& views, bool transpose_rhs = false) noexcept {
    const auto kernel = PrepareMatMulKernel(transpose_rhs);
    if (!kernel.ok()) {
        return kernel.status();
    }
    return RunMatMulEntryWith(*kernel, views);
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(CPUKernelMatMul, CpuBackendPreparesPlainFp32ReferenceKernel) {
    const auto kernel = PrepareMatMulKernel(true);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kMatMul);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::matmul_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::MatMulFp32KernelArgs));
    ASSERT_EQ(kernel->attrs.size(), 1U);
    EXPECT_EQ(std::to_integer<uint8_t>(kernel->attrs[0]), 1U);
}

TEST(CPUKernelMatMul, ReferenceExecutesRankTwoMatrices) {
    constexpr int64_t lhs_shape[2] = {2, 3};
    constexpr int64_t lhs_strides[2] = {3, 1};
    constexpr int64_t rhs_shape[2] = {3, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float lhs[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 4.0F};
    constexpr float rhs[6] = {2.0F, -1.0F, 0.5F, 3.0F, -2.0F, 1.0F};
    float output[4] = {};

    const Status status = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], -3.0F);
    EXPECT_FLOAT_EQ(output[1], 8.0F);
    EXPECT_FLOAT_EQ(output[2], -10.0F);
    EXPECT_FLOAT_EQ(output[3], 5.0F);
}

TEST(CPUKernelMatMul, ReferenceNormalizesTransposeAndStridedLayouts) {
    constexpr int64_t lhs_shape[2] = {2, 3};
    constexpr int64_t lhs_strides[2] = {7, 2};
    constexpr int64_t rhs_shape[2] = {2, 3};
    constexpr int64_t rhs_strides[2] = {5, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {8, 3};
    std::array<float, 12> lhs{};
    std::array<float, 9> rhs{};
    std::array<float, 12> output{};
    lhs[0] = 1.0F;
    lhs[2] = 2.0F;
    lhs[4] = 3.0F;
    lhs[7] = -1.0F;
    lhs[9] = 0.0F;
    lhs[11] = 4.0F;
    rhs[0] = 2.0F;
    rhs[1] = 0.5F;
    rhs[2] = -2.0F;
    rhs[5] = -1.0F;
    rhs[6] = 3.0F;
    rhs[7] = 1.0F;

    const Status status = RunMatMulEntry(
            MatMulTestViews{
                    .lhs = TensorView{lhs.data(), DataType::Float32(), lhs_shape, lhs_strides},
                    .rhs = TensorView{rhs.data(), DataType::Float32(), rhs_shape, rhs_strides},
                    .output = MutableTensorView{output.data(), DataType::Float32(), output_shape,
                                                output_strides},
            },
            true);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], -3.0F);
    EXPECT_FLOAT_EQ(output[3], 8.0F);
    EXPECT_FLOAT_EQ(output[8], -10.0F);
    EXPECT_FLOAT_EQ(output[11], 5.0F);
}

TEST(CPUKernelMatMul, ReferenceBroadcastsDifferentRankBatchDimensions) {
    constexpr int64_t lhs_shape[4] = {2, 1, 1, 2};
    constexpr int64_t lhs_strides[4] = {2, 2, 2, 1};
    constexpr int64_t rhs_shape[3] = {3, 2, 2};
    constexpr int64_t rhs_strides[3] = {4, 2, 1};
    constexpr int64_t output_shape[4] = {2, 3, 1, 2};
    constexpr int64_t output_strides[4] = {6, 2, 2, 1};
    constexpr float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float rhs[12] = {
            1.0F,
            1.0F,
            2.0F,
            2.0F,
            2.0F,
            1.0F,
            2.0F,
            3.0F,
            3.0F,
            1.0F,
            2.0F,
            4.0F,
    };
    float output[12] = {};

    const Status status = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    constexpr float expected[12] = {5.0F, 5.0F, 6.0F, 7.0F, 7.0F, 9.0F,
                                    11.0F, 11.0F, 14.0F, 15.0F, 17.0F, 19.0F};
    for (size_t i = 0; i < std::size(expected); ++i) {
        EXPECT_FLOAT_EQ(output[i], expected[i]) << "output index " << i;
    }
}

TEST(CPUKernelMatMul, ReferenceAllowsRepeatedInputElements) {
    constexpr int64_t lhs_shape[2] = {2, 2};
    constexpr int64_t lhs_strides[2] = {0, 1};
    constexpr int64_t rhs_shape[2] = {2, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float lhs[2] = {2.0F, -3.0F};
    constexpr float rhs[4] = {1.0F, 0.0F, 0.0F, 1.0F};
    float output[4] = {};

    const Status status = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 2.0F);
    EXPECT_FLOAT_EQ(output[1], -3.0F);
    EXPECT_FLOAT_EQ(output[2], 2.0F);
    EXPECT_FLOAT_EQ(output[3], -3.0F);
}

TEST(CPUKernelMatMul, ZeroInnerDimensionWritesZeroWithoutInputStorage) {
    constexpr int64_t lhs_shape[2] = {2, 0};
    constexpr int64_t lhs_strides[2] = {1, 1};
    constexpr int64_t rhs_shape[2] = {0, 3};
    constexpr int64_t rhs_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 1};
    float output[6] = {1.0F, 1.0F, 1.0F, 1.0F, 1.0F, 1.0F};

    const Status status = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{nullptr, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{nullptr, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    for (float value: output) {
        EXPECT_FLOAT_EQ(value, 0.0F);
    }
}

TEST(CPUKernelMatMul, EmptyMatrixAndBatchDimensionsAreSuccessfulNoOps) {
    constexpr int64_t lhs_empty_m_shape[2] = {0, 3};
    constexpr int64_t lhs_empty_m_strides[2] = {3, 1};
    constexpr int64_t rhs_shape[2] = {3, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_empty_m_shape[2] = {0, 2};
    constexpr int64_t output_empty_m_strides[2] = {2, 1};
    constexpr float rhs[6] = {};
    EXPECT_TRUE(RunMatMulEntry(MatMulTestViews{
                                       .lhs = TensorView{nullptr, DataType::Float32(), lhs_empty_m_shape,
                                                         lhs_empty_m_strides},
                                       .rhs = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
                                       .output = MutableTensorView{nullptr, DataType::Float32(),
                                                                   output_empty_m_shape,
                                                                   output_empty_m_strides},
                               })
                        .ok());

    constexpr int64_t rhs_empty_n_shape[2] = {3, 0};
    constexpr int64_t rhs_empty_n_strides[2] = {1, 1};
    constexpr int64_t lhs_non_empty_shape[2] = {2, 3};
    constexpr int64_t lhs_non_empty_strides[2] = {3, 1};
    constexpr int64_t output_empty_n_shape[2] = {2, 0};
    constexpr int64_t output_empty_n_strides[2] = {1, 1};
    constexpr float lhs_non_empty[6] = {};
    EXPECT_TRUE(RunMatMulEntry(MatMulTestViews{
                                       .lhs = TensorView{lhs_non_empty, DataType::Float32(), lhs_non_empty_shape,
                                                         lhs_non_empty_strides},
                                       .rhs = TensorView{nullptr, DataType::Float32(), rhs_empty_n_shape,
                                                         rhs_empty_n_strides},
                                       .output = MutableTensorView{nullptr, DataType::Float32(),
                                                                   output_empty_n_shape,
                                                                   output_empty_n_strides},
                               })
                        .ok());

    constexpr int64_t lhs_empty_batch_shape[3] = {0, 2, 3};
    constexpr int64_t lhs_empty_batch_strides[3] = {6, 3, 1};
    constexpr int64_t rhs_batched_shape[3] = {1, 3, 2};
    constexpr int64_t rhs_batched_strides[3] = {6, 2, 1};
    constexpr int64_t output_empty_batch_shape[3] = {0, 2, 2};
    constexpr int64_t output_empty_batch_strides[3] = {4, 2, 1};
    constexpr float rhs_batched[6] = {};
    EXPECT_TRUE(RunMatMulEntry(MatMulTestViews{
                                       .lhs = TensorView{nullptr, DataType::Float32(), lhs_empty_batch_shape,
                                                         lhs_empty_batch_strides},
                                       .rhs = TensorView{rhs_batched, DataType::Float32(), rhs_batched_shape,
                                                         rhs_batched_strides},
                                       .output = MutableTensorView{nullptr, DataType::Float32(),
                                                                   output_empty_batch_shape,
                                                                   output_empty_batch_strides},
                               })
                        .ok());
}

TEST(CPUKernelMatMulEntry, RejectsIncorrectOutputShape) {
    constexpr int64_t lhs_shape[2] = {2, 2};
    constexpr int64_t lhs_strides[2] = {2, 1};
    constexpr int64_t rhs_shape[2] = {2, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr int64_t incorrect_output_shape[2] = {2, 3};
    float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float rhs[4] = {1.0F, 0.0F, 0.0F, 1.0F};
    float output[6] = {};

    const Status wrong_shape = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), incorrect_output_shape, output_strides},
    });
    EXPECT_EQ(wrong_shape.code(), StatusCode::kInvalidArgument) << wrong_shape.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsOverlappingOutputLayout) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    constexpr int64_t overlapping_output_strides[2] = {1, 1};
    constexpr float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float rhs[4] = {1.0F, 0.0F, 0.0F, 1.0F};
    float output[4] = {};

    const Status overlapping_output = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), matrix_shape, matrix_strides},
            .rhs = TensorView{rhs, DataType::Float32(), matrix_shape, matrix_strides},
            .output = MutableTensorView{output, DataType::Float32(), matrix_shape,
                                        overlapping_output_strides},
    });
    EXPECT_EQ(overlapping_output.code(), StatusCode::kInvalidArgument)
            << overlapping_output.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsPartiallyAliasedLhsOutput) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    float lhs[5] = {1.0F, 2.0F, 3.0F, 4.0F, 0.0F};
    constexpr float rhs[4] = {1.0F, 0.0F, 0.0F, 1.0F};

    const Status partial_alias = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), matrix_shape, matrix_strides},
            .rhs = TensorView{rhs, DataType::Float32(), matrix_shape, matrix_strides},
            .output = MutableTensorView{lhs + 1, DataType::Float32(), matrix_shape, matrix_strides},
    });
    EXPECT_EQ(partial_alias.code(), StatusCode::kInvalidArgument) << partial_alias.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsPartiallyAliasedRhsOutput) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    constexpr float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float rhs[5] = {1.0F, 0.0F, 0.0F, 1.0F, 0.0F};

    const Status partial_alias = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{lhs, DataType::Float32(), matrix_shape, matrix_strides},
            .rhs = TensorView{rhs, DataType::Float32(), matrix_shape, matrix_strides},
            .output = MutableTensorView{rhs + 1, DataType::Float32(), matrix_shape, matrix_strides},
    });
    EXPECT_EQ(partial_alias.code(), StatusCode::kInvalidArgument) << partial_alias.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsOverflowBeforeAddressCalculation) {
    constexpr int64_t lhs_shape[2] = {2, 2};
    constexpr int64_t lhs_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    constexpr int64_t rhs_shape[2] = {2, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float storage[4] = {};
    float output[4] = {};

    const Status status = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{storage, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs = TensorView{storage, DataType::Float32(), rhs_shape, rhs_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsInvalidDtype) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    float storage[8] = {};

    const Status bad_dtype = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{storage, DataType::Double(), matrix_shape, matrix_strides},
            .rhs = TensorView{storage, DataType::Float32(), matrix_shape, matrix_strides},
            .output = MutableTensorView{storage + 4, DataType::Float32(), matrix_shape, matrix_strides},
    });
    EXPECT_EQ(bad_dtype.code(), StatusCode::kInvalidArgument) << bad_dtype.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsRankOneInput) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    constexpr int64_t rank_one_shape[1] = {2};
    constexpr int64_t rank_one_strides[1] = {1};
    float storage[8] = {};

    const Status bad_rank = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{storage, DataType::Float32(), rank_one_shape, rank_one_strides},
            .rhs = TensorView{storage, DataType::Float32(), matrix_shape, matrix_strides},
            .output = MutableTensorView{storage + 4, DataType::Float32(), matrix_shape, matrix_strides},
    });
    EXPECT_EQ(bad_rank.code(), StatusCode::kInvalidArgument) << bad_rank.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsMismatchedInnerDimension) {
    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    constexpr int64_t bad_rhs_shape[2] = {3, 2};
    float storage[12] = {};

    const Status bad_inner_dimension = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{storage, DataType::Float32(), matrix_shape, matrix_strides},
            .rhs = TensorView{storage, DataType::Float32(), bad_rhs_shape, matrix_strides},
            .output = MutableTensorView{storage + 6, DataType::Float32(), matrix_shape, matrix_strides},
    });
    EXPECT_EQ(bad_inner_dimension.code(), StatusCode::kInvalidArgument)
            << bad_inner_dimension.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsIncompatibleBatchDimensions) {
    constexpr int64_t lhs_batch_shape[3] = {2, 2, 2};
    constexpr int64_t rhs_batch_shape[3] = {3, 2, 2};
    constexpr int64_t batch_strides[3] = {4, 2, 1};
    constexpr int64_t output_batch_shape[3] = {3, 2, 2};
    float storage[12] = {};

    const Status bad_batch_shape = RunMatMulEntry(MatMulTestViews{
            .lhs = TensorView{storage, DataType::Float32(), lhs_batch_shape, batch_strides},
            .rhs = TensorView{storage, DataType::Float32(), rhs_batch_shape, batch_strides},
            .output = MutableTensorView{storage, DataType::Float32(), output_batch_shape, batch_strides},
    });
    EXPECT_EQ(bad_batch_shape.code(), StatusCode::kInvalidArgument) << bad_batch_shape.ToString();
}

TEST(CPUKernelMatMulEntry, RejectsCorruptTransposeMetadata) {
    auto kernel = PrepareMatMulKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_EQ(kernel->attrs.size(), 1U);
    kernel->attrs[0] = static_cast<std::byte>(2);

    constexpr int64_t matrix_shape[2] = {2, 2};
    constexpr int64_t matrix_strides[2] = {2, 1};
    constexpr float lhs[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float rhs[4] = {1.0F, 0.0F, 0.0F, 1.0F};
    float output[4] = {};

    const auto prepared = BuildMatMulPreparedParams(*kernel, MatMulTestViews{
                                                                     .lhs = TensorView{lhs, DataType::Float32(), matrix_shape, matrix_strides},
                                                                     .rhs = TensorView{rhs, DataType::Float32(), matrix_shape, matrix_strides},
                                                                     .output = MutableTensorView{output, DataType::Float32(), matrix_shape, matrix_strides},
                                                             });
    ASSERT_FALSE(prepared.ok());
    EXPECT_EQ(prepared.status().code(), StatusCode::kInvalidArgument)
            << prepared.status().ToString();
}

TEST(CPUKernelMatMul, ExecutionPlanBuilderRunsResolvedKernel) {
    RuntimeBuilder builder;
    Runtime runtime = builder.Build();
    const std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };
    const auto analyzed = InferOperator(OpType::kMatMul, OpParams{MatMulParams{}}, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    const std::vector<ExecutionPlanNodeSpec> nodes = {
            ExecutionPlanNodeSpec{
                    .op_type = OpType::kMatMul,
                    .selector = MakeMatMulSelector(),
                    .input_specs = inputs,
                    .output_specs = analyzed->outputs,
                    .runtime_checks = analyzed->runtime_checks,
                    .op_params = OpParams{MatMulParams{}},
            },
    };
    const auto plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    constexpr int64_t lhs_shape[2] = {2, 3};
    constexpr int64_t lhs_strides[2] = {3, 1};
    constexpr int64_t rhs_shape[2] = {3, 2};
    constexpr int64_t rhs_strides[2] = {2, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr float lhs[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.0F, 4.0F};
    constexpr float rhs[6] = {2.0F, -1.0F, 0.5F, 3.0F, -2.0F, 1.0F};
    float output[4] = {};

    test::ExecutionBindingCollector collector(*plan, runtime.GetAllocator(Device::CPU()));
    collector.Set(0, StepTensorBinding{
                             .inputs = {
                                     TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
                                     TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
                             },
                             .outputs = {
                                     MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
                             },
                     });
    auto bindings = collector.CreateContext();
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *bindings).ok());
    EXPECT_FLOAT_EQ(output[0], -3.0F);
    EXPECT_FLOAT_EQ(output[1], 8.0F);
    EXPECT_FLOAT_EQ(output[2], -10.0F);
    EXPECT_FLOAT_EQ(output[3], 5.0F);
}

} // namespace
