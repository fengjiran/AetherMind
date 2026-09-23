#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_weight_prepacker.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/compiler/graph_lowering.h"
#include "aethermind/compiler/optimize_graph.h"
#include "aethermind/compiler/packing_request_builder.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/graph/graph.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight/weight_packing.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/add_rmsnorm/add_rmsnorm_internal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

constexpr float kEpsilon = 1.0e-5F;

KernelSelector MakeAddRmsNormSelector(WeightFormat weight_format) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = weight_format,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareAddRmsNormKernel(
        WeightFormat weight_format = WeightFormat::kPlain,
        float epsilon = kEpsilon) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kAddRmsNorm, MakeAddRmsNormSelector(weight_format),
            OpParams{AddRmsNormParams{.eps = epsilon}});
}

struct AddRmsNormTestViews {
    TensorView input{};
    TensorView residual{};
    TensorView weight{};
    MutableTensorView output{};
    MutableTensorView new_residual{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

PackedWeightView MakeIdentityPackedWeight(const float* data,
                                          size_t nbytes,
                                          std::span<const int64_t> shape) {
    return PackedWeightView{
            .data = data,
            .nbytes = nbytes,
            .logical_dtype = DataType::Float32(),
            .logical_shape = shape,
            .recipe_layout = cpu::kCpuIdentityPackingLayout,
            .recipe_alignment = cpu::kCpuIdentityPackingAlignment,
            .alignment = cpu::kCpuIdentityPackingAlignment,
    };
}

StatusOr<PreparedKernelParams> BuildPlainPreparedParams(
        const ResolvedKernel& kernel,
        const AddRmsNormTestViews& views) {
    const std::array<TensorView, 3> inputs = {
            views.input, views.residual, views.weight};
    const std::array<MutableTensorView, 2> outputs = {
            views.output, views.new_residual};
    PreparedKernelParams prepared;
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
            },
            prepared.storage.data()));
    return prepared;
}

StatusOr<PreparedKernelParams> BuildPackedPreparedParams(
        const ResolvedKernel& kernel,
        const AddRmsNormTestViews& views,
        PackedWeightView packed_weight) {
    const std::array<TensorView, 2> inputs = {views.input, views.residual};
    const std::array<MutableTensorView, 2> outputs = {
            views.output, views.new_residual};
    PreparedKernelParams prepared;
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
                    .packed_weight = packed_weight,
            },
            prepared.storage.data()));
    return prepared;
}

Status RunAddRmsNormEntry(const ResolvedKernel& kernel,
                          const PreparedKernelParams& prepared) {
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

Status RunPlainAddRmsNorm(const ResolvedKernel& kernel,
                          const AddRmsNormTestViews& views) {
    AM_ASSIGN_OR_RETURN(const PreparedKernelParams prepared,
                        BuildPlainPreparedParams(kernel, views));
    return RunAddRmsNormEntry(kernel, prepared);
}

Status RunPackedAddRmsNorm(const ResolvedKernel& kernel,
                           const AddRmsNormTestViews& views,
                           PackedWeightView packed_weight) {
    AM_ASSIGN_OR_RETURN(const PreparedKernelParams prepared,
                        BuildPackedPreparedParams(kernel, views, packed_weight));
    return RunAddRmsNormEntry(kernel, prepared);
}

void ExpectRowsNear(const float* input,
                    const float* residual,
                    const float* weight,
                    const float* output,
                    const float* new_residual,
                    int64_t row_count,
                    int64_t hidden_size,
                    int64_t input_row_stride,
                    int64_t input_col_stride,
                    int64_t residual_row_stride,
                    int64_t residual_col_stride,
                    int64_t weight_stride,
                    int64_t output_row_stride,
                    int64_t output_col_stride,
                    int64_t new_residual_row_stride,
                    int64_t new_residual_col_stride,
                    float epsilon = kEpsilon) {
    for (int64_t row = 0; row < row_count; ++row) {
        const float* const input_row = input + row * input_row_stride;
        const float* const residual_row = residual + row * residual_row_stride;
        const float* const output_row = output + row * output_row_stride;
        const float* const new_residual_row =
                new_residual + row * new_residual_row_stride;

        double sum_sq = 0.0;
        for (int64_t col = 0; col < hidden_size; ++col) {
            const float expected_sum = input_row[col * input_col_stride] +
                                       residual_row[col * residual_col_stride];
            EXPECT_FLOAT_EQ(
                    new_residual_row[col * new_residual_col_stride], expected_sum)
                    << "new_residual mismatch at row " << row << ", col " << col;
            const double value = static_cast<double>(expected_sum);
            sum_sq += value * value;
        }

        const double inv_rms = 1.0 / std::sqrt(
                                             sum_sq / static_cast<double>(hidden_size) +
                                             static_cast<double>(epsilon));
        for (int64_t col = 0; col < hidden_size; ++col) {
            const float expected_sum = input_row[col * input_col_stride] +
                                       residual_row[col * residual_col_stride];
            const float expected_output = static_cast<float>(
                    static_cast<double>(expected_sum) * inv_rms *
                    static_cast<double>(weight[col * weight_stride]));
            EXPECT_NEAR(output_row[col * output_col_stride], expected_output, 1.0e-6F)
                    << "output mismatch at row " << row << ", col " << col;
        }
    }
}

TEST(CPUKernelAddRmsNorm, ResolvesPlainAndPackedIdentityDescriptors) {
    const auto plain = PrepareAddRmsNormKernel(WeightFormat::kPlain);
    ASSERT_TRUE(plain.ok()) << plain.status().ToString();
    EXPECT_EQ(std::string_view(plain->name), "cpu::add_rmsnorm_f32_reference");
    EXPECT_EQ(plain->params_size, sizeof(cpu::detail::AddRmsNormF32KernelArgs));
    EXPECT_NE(plain->fn, nullptr);
    EXPECT_NE(plain->params_builder, nullptr);

    const auto packed = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ(std::string_view(packed->name),
              "cpu::add_rmsnorm_f32_packed_identity_reference");
    EXPECT_EQ(packed->params_size, sizeof(cpu::detail::AddRmsNormF32KernelArgs));
    EXPECT_NE(packed->fn, nullptr);
    EXPECT_NE(packed->params_builder, nullptr);
    EXPECT_FALSE(PrepareAddRmsNormKernel(WeightFormat::kPlain, 0.0F).ok());
}

TEST(CPUKernelAddRmsNorm, PlainReferenceSupportsPaddedAndNonUnitStrides) {
    constexpr std::array<int64_t, 2> shape{2, 3};
    constexpr std::array<int64_t, 2> input_strides{8, 2};
    constexpr std::array<int64_t, 2> residual_strides{9, 2};
    constexpr std::array<int64_t, 2> output_strides{10, 2};
    constexpr std::array<int64_t, 2> new_residual_strides{11, 3};
    constexpr std::array<int64_t, 1> weight_shape{3};
    constexpr std::array<int64_t, 1> weight_strides{2};
    alignas(64) std::array<float, 16> input{};
    alignas(64) std::array<float, 18> residual{};
    alignas(64) std::array<float, 20> output{};
    alignas(64) std::array<float, 22> new_residual{};
    alignas(64) const std::array<float, 5> weight{1.0F, 0.0F, 0.5F, 0.0F, -1.5F};
    input[0] = 1.0F;
    input[2] = -2.0F;
    input[4] = 3.0F;
    input[8] = -1.0F;
    input[10] = 0.5F;
    input[12] = 4.0F;
    residual[0] = 0.5F;
    residual[2] = 1.0F;
    residual[4] = -2.0F;
    residual[9] = 2.0F;
    residual[11] = -0.5F;
    residual[13] = -1.0F;

    const auto kernel = PrepareAddRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunPlainAddRmsNorm(
                        *kernel,
                        {.input = TensorView(input.data(), DataType::Float32(), shape,
                                             input_strides, 64),
                         .residual = TensorView(residual.data(), DataType::Float32(), shape,
                                                residual_strides, 64),
                         .weight = TensorView(weight.data(), DataType::Float32(), weight_shape,
                                              weight_strides, 64),
                         .output = MutableTensorView(output.data(), DataType::Float32(), shape,
                                                     output_strides, 64),
                         .new_residual = MutableTensorView(new_residual.data(), DataType::Float32(),
                                                           shape, new_residual_strides, 64)})
                        .ok());

    ExpectRowsNear(input.data(), residual.data(), weight.data(), output.data(),
                   new_residual.data(), 2, 3, 8, 2, 9, 2, 2, 10, 2, 11, 3);
}

TEST(CPUKernelAddRmsNorm, PlainReferenceSupportsRankOneThreeAndFour) {
    const auto kernel = PrepareAddRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr std::array<int64_t, 1> rank_one_shape{3};
    constexpr std::array<int64_t, 1> rank_one_strides{1};
    constexpr std::array<float, 3> rank_one_input{1.0F, -2.0F, 3.0F};
    constexpr std::array<float, 3> rank_one_residual{0.5F, 1.0F, -2.0F};
    constexpr std::array<float, 3> weight{1.0F, 0.5F, 2.0F};
    std::array<float, 3> rank_one_output{};
    std::array<float, 3> rank_one_new_residual{};
    ASSERT_TRUE(RunPlainAddRmsNorm(
                        *kernel,
                        {.input = TensorView(rank_one_input.data(), DataType::Float32(),
                                             rank_one_shape, rank_one_strides),
                         .residual = TensorView(rank_one_residual.data(), DataType::Float32(),
                                                rank_one_shape, rank_one_strides),
                         .weight = TensorView(weight.data(), DataType::Float32(),
                                              rank_one_shape, rank_one_strides),
                         .output = MutableTensorView(rank_one_output.data(), DataType::Float32(),
                                                     rank_one_shape, rank_one_strides),
                         .new_residual = MutableTensorView(rank_one_new_residual.data(),
                                                           DataType::Float32(), rank_one_shape,
                                                           rank_one_strides)})
                        .ok());
    ExpectRowsNear(rank_one_input.data(), rank_one_residual.data(), weight.data(),
                   rank_one_output.data(), rank_one_new_residual.data(), 1, 3, 0, 1, 0, 1,
                   1, 0, 1, 0, 1);

    constexpr std::array<int64_t, 3> rank_three_shape{2, 2, 3};
    constexpr std::array<int64_t, 3> rank_three_strides{6, 3, 1};
    std::array<float, 12> rank_three_input{};
    std::array<float, 12> rank_three_residual{};
    std::array<float, 12> rank_three_output{};
    std::array<float, 12> rank_three_new_residual{};
    for (int64_t index = 0; index < 12; ++index) {
        rank_three_input[static_cast<size_t>(index)] = static_cast<float>(index - 6);
        rank_three_residual[static_cast<size_t>(index)] = static_cast<float>(3 - index) * 0.25F;
    }
    ASSERT_TRUE(RunPlainAddRmsNorm(
                        *kernel,
                        {.input = TensorView(rank_three_input.data(), DataType::Float32(),
                                             rank_three_shape, rank_three_strides),
                         .residual = TensorView(rank_three_residual.data(), DataType::Float32(),
                                                rank_three_shape, rank_three_strides),
                         .weight = TensorView(weight.data(), DataType::Float32(),
                                              rank_one_shape, rank_one_strides),
                         .output = MutableTensorView(rank_three_output.data(), DataType::Float32(),
                                                     rank_three_shape, rank_three_strides),
                         .new_residual = MutableTensorView(rank_three_new_residual.data(),
                                                           DataType::Float32(), rank_three_shape,
                                                           rank_three_strides)})
                        .ok());
    ExpectRowsNear(rank_three_input.data(), rank_three_residual.data(), weight.data(),
                   rank_three_output.data(), rank_three_new_residual.data(), 4, 3, 3, 1, 3, 1,
                   1, 3, 1, 3, 1);

    constexpr std::array<int64_t, 4> rank_four_shape{2, 1, 2, 3};
    constexpr std::array<int64_t, 4> rank_four_strides{6, 6, 3, 1};
    std::array<float, 12> rank_four_output{};
    std::array<float, 12> rank_four_new_residual{};
    ASSERT_TRUE(RunPlainAddRmsNorm(
                        *kernel,
                        {.input = TensorView(rank_three_input.data(), DataType::Float32(),
                                             rank_four_shape, rank_four_strides),
                         .residual = TensorView(rank_three_residual.data(), DataType::Float32(),
                                                rank_four_shape, rank_four_strides),
                         .weight = TensorView(weight.data(), DataType::Float32(),
                                              rank_one_shape, rank_one_strides),
                         .output = MutableTensorView(rank_four_output.data(), DataType::Float32(),
                                                     rank_four_shape, rank_four_strides),
                         .new_residual = MutableTensorView(rank_four_new_residual.data(),
                                                           DataType::Float32(), rank_four_shape,
                                                           rank_four_strides)})
                        .ok());
    ExpectRowsNear(rank_three_input.data(), rank_three_residual.data(), weight.data(),
                   rank_four_output.data(), rank_four_new_residual.data(), 4, 3, 3, 1, 3, 1,
                   1, 3, 1, 3, 1);
}

TEST(CPUKernelAddRmsNorm, PackedIdentityReferencePreservesRoundedAddition) {
    constexpr std::array<int64_t, 2> shape{1, 2};
    constexpr std::array<int64_t, 2> strides{2, 1};
    constexpr std::array<int64_t, 1> weight_shape{2};
    alignas(64) const float input[2] = {16777216.0F, 1.0F};
    alignas(64) const float residual[2] = {1.0F, -0.5F};
    alignas(64) const float packed_weight[2] = {1.0F, -2.0F};
    alignas(64) float output[2]{};
    alignas(64) float new_residual[2]{};

    const auto kernel = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunPackedAddRmsNorm(
                        *kernel,
                        {.input = TensorView(input, DataType::Float32(), shape, strides, 64),
                         .residual = TensorView(residual, DataType::Float32(), shape, strides, 64),
                         .output = MutableTensorView(output, DataType::Float32(), shape, strides, 64),
                         .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape,
                                                           strides, 64)},
                        MakeIdentityPackedWeight(packed_weight, sizeof(packed_weight),
                                                 weight_shape))
                        .ok());

    // `16777216 + 1` rounds back to 16777216 in Float32. The reference
    // kernel must normalize this stored Add result, not a widened addition.
    EXPECT_FLOAT_EQ(new_residual[0], input[0]);
    ExpectRowsNear(input, residual, packed_weight, output, new_residual, 1, 2, 2, 1, 2, 1,
                   1, 2, 1, 2, 1);
}

TEST(CPUKernelAddRmsNorm, ZeroRowsValidateMetadataButNeedNoStorage) {
    constexpr std::array<int64_t, 2> activation_shape{0, 3};
    constexpr std::array<int64_t, 2> activation_strides{3, 1};
    constexpr std::array<int64_t, 1> weight_shape{3};
    constexpr std::array<int64_t, 1> weight_strides{1};
    alignas(64) constexpr float plain_weight[3] = {1.0F, 1.0F, 1.0F};
    const AddRmsNormTestViews views{
            .input = TensorView(nullptr, DataType::Float32(), activation_shape, activation_strides),
            .residual = TensorView(nullptr, DataType::Float32(), activation_shape,
                                   activation_strides),
            .weight = TensorView(plain_weight, DataType::Float32(), weight_shape, weight_strides,
                                 64),
            .output = MutableTensorView(nullptr, DataType::Float32(), activation_shape,
                                        activation_strides),
            .new_residual = MutableTensorView(nullptr, DataType::Float32(), activation_shape,
                                              activation_strides),
    };

    const auto plain = PrepareAddRmsNormKernel();
    ASSERT_TRUE(plain.ok()) << plain.status().ToString();
    EXPECT_TRUE(RunPlainAddRmsNorm(*plain, views).ok());

    const auto packed = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_TRUE(RunPackedAddRmsNorm(
                        *packed, views,
                        MakeIdentityPackedWeight(nullptr, 3 * sizeof(float), weight_shape))
                        .ok());

    PackedWeightView wrong_recipe =
            MakeIdentityPackedWeight(nullptr, 3 * sizeof(float), weight_shape);
    wrong_recipe.recipe_layout = "different_recipe";
    EXPECT_EQ(BuildPackedPreparedParams(*packed, views, wrong_recipe).status().code(),
              StatusCode::kInvalidArgument);

    const size_t excessive_nbytes =
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) + size_t{1};
    EXPECT_EQ(BuildPackedPreparedParams(
                      *packed, views,
                      MakeIdentityPackedWeight(nullptr, excessive_nbytes, weight_shape))
                      .status()
                      .code(),
              StatusCode::kOverflow);
}

TEST(CPUKernelAddRmsNorm, RejectsInvalidMetadataShapesLayoutsAndPointers) {
    constexpr std::array<int64_t, 2> shape{1, 3};
    constexpr std::array<int64_t, 2> strides{3, 1};
    constexpr std::array<int64_t, 1> weight_shape{3};
    constexpr std::array<int64_t, 1> weight_strides{1};
    alignas(64) float input[6]{};
    alignas(64) float residual[6]{};
    alignas(64) float weight[3]{};
    alignas(64) float output[6]{};
    alignas(64) float new_residual[6]{};
    const AddRmsNormTestViews valid{
            .input = TensorView(input, DataType::Float32(), shape, strides, 64),
            .residual = TensorView(residual, DataType::Float32(), shape, strides, 64),
            .weight = TensorView(weight, DataType::Float32(), weight_shape, weight_strides, 64),
            .output = MutableTensorView(output, DataType::Float32(), shape, strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape, strides,
                                              64),
    };
    const auto kernel = PrepareAddRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    PreparedKernelParams prepared;
    const std::array<TensorView, 3> inputs{valid.input, valid.residual, valid.weight};
    const std::array<MutableTensorView, 2> outputs{valid.output, valid.new_residual};
    EXPECT_EQ(kernel->params_builder(
                            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs},
                            prepared.storage.data())
                      .code(),
              StatusCode::kInvalidArgument);

    AddRmsNormTestViews wrong_dtype = valid;
    wrong_dtype.residual = TensorView(residual, DataType::Float(16), shape, strides, 64);
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, wrong_dtype).status().code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 2> wrong_shape{1, 2};
    AddRmsNormTestViews wrong_activation_shape = valid;
    wrong_activation_shape.residual = TensorView(residual, DataType::Float32(), wrong_shape,
                                                 strides, 64);
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, wrong_activation_shape).status().code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 2> rank_two_weight_shape{1, 3};
    AddRmsNormTestViews wrong_weight = valid;
    wrong_weight.weight = TensorView(weight, DataType::Float32(), rank_two_weight_shape,
                                     strides, 64);
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, wrong_weight).status().code(),
              StatusCode::kInvalidArgument);

    // TensorView forbids constructing a non-empty null-data view, so this
    // represents the only untrusted null/default view reaching the builder.
    AddRmsNormTestViews invalid_input = valid;
    invalid_input.input = TensorView{};
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, invalid_input).status().code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 3> noncollapsible_shape{2, 2, 3};
    constexpr std::array<int64_t, 3> noncollapsible_strides{7, 3, 1};
    AddRmsNormTestViews noncollapsible{
            .input = TensorView(input, DataType::Float32(), noncollapsible_shape,
                                noncollapsible_strides, 64),
            .residual = TensorView(residual, DataType::Float32(), noncollapsible_shape,
                                   noncollapsible_strides, 64),
            .weight = valid.weight,
            .output = MutableTensorView(output, DataType::Float32(), noncollapsible_shape,
                                        noncollapsible_strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(),
                                              noncollapsible_shape, noncollapsible_strides, 64),
    };
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, noncollapsible).status().code(),
              StatusCode::kUnimplemented);

    constexpr std::array<int64_t, 2> two_row_shape{2, 3};
    constexpr std::array<int64_t, 2> overlapping_output_strides{1, 1};
    AddRmsNormTestViews overlapping_rows = valid;
    overlapping_rows.output = MutableTensorView(output, DataType::Float32(),
                                                two_row_shape,
                                                overlapping_output_strides, 64);
    overlapping_rows.input = TensorView(input, DataType::Float32(), two_row_shape, strides, 64);
    overlapping_rows.residual =
            TensorView(residual, DataType::Float32(), two_row_shape, strides, 64);
    overlapping_rows.new_residual = MutableTensorView(new_residual, DataType::Float32(),
                                                      two_row_shape, strides, 64);
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, overlapping_rows).status().code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 3> overflowing_shape{
            std::numeric_limits<int64_t>::max(), 2, 1};
    constexpr std::array<int64_t, 3> overflowing_strides{2, 1, 1};
    constexpr std::array<int64_t, 1> single_weight_shape{1};
    AddRmsNormTestViews overflowing_rows{
            .input = TensorView(input, DataType::Float32(), overflowing_shape,
                                overflowing_strides, 64),
            .residual = TensorView(residual, DataType::Float32(), overflowing_shape,
                                   overflowing_strides, 64),
            .weight = TensorView(weight, DataType::Float32(), single_weight_shape,
                                 weight_strides, 64),
            .output = MutableTensorView(output, DataType::Float32(), overflowing_shape,
                                        overflowing_strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(),
                                              overflowing_shape, overflowing_strides, 64),
    };
    EXPECT_EQ(BuildPlainPreparedParams(*kernel, overflowing_rows).status().code(),
              StatusCode::kInvalidArgument);

    const auto packed_kernel = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(packed_kernel.ok()) << packed_kernel.status().ToString();
    EXPECT_EQ(BuildPackedPreparedParams(
                      *packed_kernel, valid,
                      MakeIdentityPackedWeight(nullptr, sizeof(weight), weight_shape))
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelAddRmsNorm, RejectsPlainAndPackedOutputAliases) {
    constexpr std::array<int64_t, 2> shape{1, 3};
    constexpr std::array<int64_t, 2> strides{3, 1};
    constexpr std::array<int64_t, 1> weight_shape{3};
    constexpr std::array<int64_t, 1> weight_strides{1};
    alignas(64) float input[3]{};
    alignas(64) float residual[3]{};
    alignas(64) float weight[3]{};
    alignas(64) float output[3]{};
    alignas(64) float new_residual[3]{};
    const auto plain = PrepareAddRmsNormKernel();
    ASSERT_TRUE(plain.ok()) << plain.status().ToString();

    AddRmsNormTestViews plain_alias{
            .input = TensorView(input, DataType::Float32(), shape, strides, 64),
            .residual = TensorView(residual, DataType::Float32(), shape, strides, 64),
            .weight = TensorView(weight, DataType::Float32(), weight_shape, weight_strides, 64),
            .output = MutableTensorView(output, DataType::Float32(), shape, strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape, strides,
                                              64),
    };
    const auto expect_plain_alias_rejected = [&](float* output_data, float* new_residual_data) {
        plain_alias.output = MutableTensorView(output_data, DataType::Float32(), shape, strides, 64);
        plain_alias.new_residual =
                MutableTensorView(new_residual_data, DataType::Float32(), shape, strides, 64);
        EXPECT_EQ(BuildPlainPreparedParams(*plain, plain_alias).status().code(),
                  StatusCode::kInvalidArgument);
    };
    expect_plain_alias_rejected(input, new_residual);    // output/input
    expect_plain_alias_rejected(residual, new_residual); // output/residual
    expect_plain_alias_rejected(weight, new_residual);   // output/weight
    expect_plain_alias_rejected(output, input);          // new_residual/input
    expect_plain_alias_rejected(output, residual);       // new_residual/residual
    expect_plain_alias_rejected(output, weight);         // new_residual/weight
    expect_plain_alias_rejected(output, output);         // output/new_residual

    const auto packed = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    alignas(64) float packed_weight[3]{};
    AddRmsNormTestViews packed_output_alias{
            .input = TensorView(input, DataType::Float32(), shape, strides, 64),
            .residual = TensorView(residual, DataType::Float32(), shape, strides, 64),
            .output = MutableTensorView(packed_weight, DataType::Float32(), shape, strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape, strides,
                                              64),
    };
    EXPECT_EQ(BuildPackedPreparedParams(
                      *packed, packed_output_alias,
                      MakeIdentityPackedWeight(packed_weight, sizeof(packed_weight), weight_shape))
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);

    packed_output_alias.output = MutableTensorView(output, DataType::Float32(), shape, strides,
                                                   64);
    packed_output_alias.new_residual = MutableTensorView(
            packed_weight, DataType::Float32(), shape, strides, 64);
    EXPECT_EQ(BuildPackedPreparedParams(
                      *packed, packed_output_alias,
                      MakeIdentityPackedWeight(packed_weight, sizeof(packed_weight), weight_shape))
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelAddRmsNorm, ReportsUnimplementedForPackedWeightInOutputStrideHole) {
    constexpr std::array<int64_t, 2> shape{1, 2};
    constexpr std::array<int64_t, 2> contiguous_strides{2, 1};
    constexpr std::array<int64_t, 2> output_strides{3, 2};
    constexpr std::array<int64_t, 1> weight_shape{2};
    alignas(64) float input[2]{};
    alignas(64) float residual[2]{};
    alignas(64) float output_storage[4]{};
    alignas(64) float new_residual[2]{};
    const auto kernel = PrepareAddRmsNormKernel(WeightFormat::kPacked);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const Status status = BuildPackedPreparedParams(
                                  *kernel,
                                  {.input = TensorView(input, DataType::Float32(), shape, contiguous_strides, 64),
                                   .residual = TensorView(residual, DataType::Float32(), shape, contiguous_strides, 64),
                                   .output = MutableTensorView(output_storage, DataType::Float32(), shape,
                                                               output_strides, 64),
                                   .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape,
                                                                     contiguous_strides, 64)},
                                  MakeIdentityPackedWeight(output_storage + 1, 2 * sizeof(float), weight_shape))
                                  .status();
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
}

TEST(CPUKernelAddRmsNorm, PreparedParamsExecuteRepeatedlyWithoutRevalidation) {
    constexpr std::array<int64_t, 2> shape{1, 3};
    constexpr std::array<int64_t, 2> strides{3, 1};
    constexpr std::array<int64_t, 1> weight_shape{3};
    constexpr std::array<int64_t, 1> weight_strides{1};
    alignas(64) float input[3]{};
    alignas(64) float residual[3]{};
    alignas(64) constexpr float weight[3] = {1.0F, 0.5F, 2.0F};
    alignas(64) float output[3]{};
    alignas(64) float new_residual[3]{};
    const AddRmsNormTestViews views{
            .input = TensorView(input, DataType::Float32(), shape, strides, 64),
            .residual = TensorView(residual, DataType::Float32(), shape, strides, 64),
            .weight = TensorView(weight, DataType::Float32(), weight_shape, weight_strides, 64),
            .output = MutableTensorView(output, DataType::Float32(), shape, strides, 64),
            .new_residual = MutableTensorView(new_residual, DataType::Float32(), shape, strides,
                                              64),
    };
    const auto kernel = PrepareAddRmsNormKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const auto prepared = BuildPlainPreparedParams(*kernel, views);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    constexpr float inputs[][3] = {
            {1.0F, 2.0F, 3.0F},
            {-1.0F, 0.5F, -2.0F},
            {8.0F, -8.0F, 4.0F},
    };
    constexpr float residuals[][3] = {
            {0.5F, -1.0F, 2.0F},
            {2.0F, -0.25F, 1.0F},
            {-4.0F, 4.0F, -2.0F},
    };
    for (size_t batch = 0; batch < std::size(inputs); ++batch) {
        std::copy_n(inputs[batch], 3, input);
        std::copy_n(residuals[batch], 3, residual);
        ASSERT_TRUE(RunAddRmsNormEntry(*kernel, *prepared).ok());
        ExpectRowsNear(input, residual, weight, output, new_residual, 1, 3, 3, 1, 3, 1, 1,
                       3, 1, 3, 1);
    }
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return SymbolicShape(IntArrayView{copied});
}

class FloatRawStorage final : public RawStorage {
public:
    std::vector<float> values{};
};

RawWeightView MakeRawWeight(const std::shared_ptr<FloatRawStorage>& storage,
                            std::initializer_list<int64_t> shape) {
    return RawWeightView{
            .data = reinterpret_cast<const std::byte*>(storage->values.data()),
            .bytes = storage->values.size() * sizeof(float),
            .dtype = DataType::Float32(),
            .shape = std::vector<int64_t>(shape),
            .storage = storage,
            .is_contiguous = true,
    };
}

TEST(CPUKernelAddRmsNorm, FusedPackedExecutionRunsThroughLoweringAndBindings) {
    ModelGraph graph;
    const TensorSpec activation_spec{
            .dtype = DataType::Float32(),
            .shape = StaticShape({2, 3}),
    };
    const GraphValueId input_lhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const GraphValueId input_rhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const GraphValueId residual_lhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const GraphValueId residual_rhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const auto input = graph.AddNode(
            OpType::kAdd, 0U, {input_lhs, input_rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    ASSERT_TRUE(input.ok()) << input.status().ToString();
    const auto residual = graph.AddNode(
            OpType::kAdd, 0U, {residual_lhs, residual_rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    ASSERT_TRUE(residual.ok()) << residual.status().ToString();
    const auto add = graph.AddNode(
            OpType::kAdd, 0U, {input->outputs[0], residual->outputs[0]},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    ASSERT_TRUE(add.ok()) << add.status().ToString();
    const GraphValueId weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
            MakeTransformerWeightBinding(0U, TransformerWeightRole::kInputNorm));
    const auto rmsnorm = graph.AddNode(
            OpType::kRmsNorm, 0U, {add->outputs[0], weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, RmsNormParams{.eps = kEpsilon});
    ASSERT_TRUE(rmsnorm.ok()) << rmsnorm.status().ToString();
    graph.MarkOutput(rmsnorm->outputs[0]);
    graph.MarkOutput(add->outputs[0]);

    PassContext optimization;
    optimization.const_eval_policy.max_output_bytes = 0;
    const auto optimized = OptimizeModelGraph(graph, optimization);
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    ASSERT_EQ(optimized->FindNodesByOpType(OpType::kAddRmsNorm).size(), 1U);
    const auto lowered = LowerModelGraph(
            *optimized, GraphLoweringConfig{.enable_packed_weights = true});
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto raw_storage = std::make_shared<FloatRawStorage>();
    raw_storage->values = {1.0F, 0.5F, -1.5F};
    ResolvedModelWeights resolved;
    resolved.layers.resize(1);
    resolved.layers[0].norm.input_rmsnorm = MakeRawWeight(raw_storage, {3});
    const auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    ASSERT_EQ(requests->size(), 1U);
    EXPECT_EQ(requests->front().op_type, OpType::kAddRmsNorm);
    EXPECT_EQ(requests->front().selector.weight_format, WeightFormat::kPacked);

    PackedWeightStore packed_store;
    RuntimeBuilder runtime_builder;
    runtime_builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    Runtime runtime = runtime_builder.Build();
    const auto prepack_backend = runtime.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(prepack_backend.ok()) << prepack_backend.status().ToString();
    ASSERT_TRUE(PrepackWeightRequests(**prepack_backend, packed_store, *requests).ok());
    const auto plan = ExecutionPlanBuilder::Build(runtime, packed_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 3U);
    std::array<const ExecutionStep*, 2> bridge_steps{};
    const ExecutionStep* fused_step = nullptr;
    size_t bridge_count = 0;
    for (const ExecutionStep& step: plan->steps()) {
        if (step.kernel.op_type == OpType::kAdd) {
            ASSERT_LT(bridge_count, bridge_steps.size());
            bridge_steps[bridge_count++] = &step;
        } else if (step.kernel.op_type == OpType::kAddRmsNorm) {
            fused_step = &step;
        }
    }
    ASSERT_EQ(bridge_count, bridge_steps.size());
    ASSERT_NE(fused_step, nullptr);
    EXPECT_EQ(fused_step->kernel_input_ports, std::vector<uint32_t>({0, 1}));
    ASSERT_NE(fused_step->packed_weights, nullptr);

    alignas(64) const float input_values[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F};
    alignas(64) const float residual_values[6] = {0.5F, -1.0F, 2.0F, 1.0F, -0.5F, -1.0F};
    alignas(64) const float zeros[6]{};
    alignas(64) float output_values[6]{};
    alignas(64) float new_residual_values[6]{};
    constexpr std::array<int64_t, 2> activation_shape{2, 3};
    constexpr std::array<int64_t, 2> activation_strides{3, 1};
    auto prepared = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = bridge_steps[0]->inputs[0],
                           .tensor = TensorView(input_values, DataType::Float32(), activation_shape,
                                                activation_strides, 64)},
                          {.value = bridge_steps[0]->inputs[1],
                           .tensor = TensorView(zeros, DataType::Float32(), activation_shape,
                                                activation_strides, 64)},
                          {.value = bridge_steps[1]->inputs[0],
                           .tensor = TensorView(residual_values, DataType::Float32(), activation_shape,
                                                activation_strides, 64)},
                          {.value = bridge_steps[1]->inputs[1],
                           .tensor = TensorView(zeros, DataType::Float32(), activation_shape,
                                                activation_strides, 64)}},
             .writable = {{.value = fused_step->outputs[0],
                           .tensor = MutableTensorView(output_values, DataType::Float32(),
                                                       activation_shape, activation_strides, 64)},
                          {.value = fused_step->outputs[1],
                           .tensor = MutableTensorView(new_residual_values, DataType::Float32(),
                                                       activation_shape, activation_strides, 64)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    ExpectRowsNear(input_values, residual_values, raw_storage->values.data(), output_values,
                   new_residual_values, 2, 3, 3, 1, 3, 1, 1, 3, 1, 3, 1);
}

} // namespace
