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
#include "aethermind/model/weight/weight_packing.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/gate_up_linear/gate_up_linear_internal.h"

#include <array>
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeGateUpSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareGateUpKernel(int64_t gate_features,
                                             int64_t up_features,
                                             bool has_bias = false) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kGateUpLinear, MakeGateUpSelector(),
            OpParams{GateUpLinearParams{.gate_out_features = gate_features,
                                        .up_out_features = up_features,
                                        .has_bias = has_bias}});
}

struct GateUpTestViews {
    TensorView input{};
    MutableTensorView gate{};
    MutableTensorView up{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunGateUpEntry(const ResolvedKernel& kernel,
                      const GateUpTestViews& views,
                      PackedWeightView packed) {
    const std::array<TensorView, 1> inputs{views.input};
    const std::array<MutableTensorView, 2> outputs{views.gate, views.up};
    PreparedKernelParams prepared;
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel.attrs,
                    .packed_weight = packed,
            },
            prepared.storage.data()));
    return kernel.fn(KernelContext{
            .kernel_params = prepared.storage.data(),
            .attrs = kernel.attrs,
    });
}

PackedWeightView MakeIdentityPackedWeight(const float* data,
                                          size_t nbytes,
                                          std::span<const int64_t> shape) {
    return PackedWeightView{
            .data = data,
            .nbytes = nbytes,
            .logical_dtype = DataType::Float32(),
            .logical_shape = shape,
            .recipe_layout = "cpu_identity",
            .recipe_alignment = 64,
            .alignment = 64,
    };
}

TEST(CPUKernelGateUpLinear, CpuBackendPreparesPackedF32ReferenceKernel) {
    const auto kernel = PrepareGateUpKernel(4, 7);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kGateUpLinear);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::gate_up_linear_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::GateUpLinearF32KernelArgs));

    CpuBackend backend;
    KernelSelector plain = MakeGateUpSelector();
    plain.weight_format = WeightFormat::kPlain;
    EXPECT_EQ(backend.PrepareKernel(
                             OpType::kGateUpLinear, plain,
                             OpParams{GateUpLinearParams{.gate_out_features = 4,
                                                         .up_out_features = 7}})
                      .status()
                      .code(),
              StatusCode::kNotFound);
    EXPECT_EQ(PrepareGateUpKernel(4, 7, true).status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(PrepareGateUpKernel(std::numeric_limits<int64_t>::max(), 1)
                      .status()
                      .code(),
              StatusCode::kOverflow);
}

TEST(CPUKernelGateUpLinear, ReferenceSupportsRankOneAndStridedRankThreeViews) {
    constexpr int64_t rank_one_input_shape[1] = {3};
    constexpr int64_t rank_one_input_strides[1] = {1};
    constexpr int64_t rank_one_gate_shape[1] = {2};
    constexpr int64_t rank_one_gate_strides[1] = {1};
    constexpr int64_t rank_one_up_shape[1] = {3};
    constexpr int64_t rank_one_up_strides[1] = {1};
    constexpr std::array<int64_t, 2> rank_one_packed_shape{5, 3};
    alignas(64) const float rank_one_input[3] = {1.0F, -2.0F, 0.5F};
    alignas(64) const float rank_one_packed[15] = {
            1.0F, 0.0F, -1.0F, 0.5F, 1.0F, 0.0F, -1.0F, 0.5F, 1.0F,
            2.0F, 0.0F, 0.0F, 0.0F, 2.0F, 0.0F};
    alignas(64) float rank_one_gate[2]{};
    alignas(64) float rank_one_up[3]{};

    const auto rank_one_kernel = PrepareGateUpKernel(2, 3);
    ASSERT_TRUE(rank_one_kernel.ok()) << rank_one_kernel.status().ToString();
    ASSERT_TRUE(RunGateUpEntry(
                        *rank_one_kernel,
                        {.input = TensorView(rank_one_input, DataType::Float32(), rank_one_input_shape,
                                             rank_one_input_strides, 64),
                         .gate = MutableTensorView(rank_one_gate, DataType::Float32(),
                                                   rank_one_gate_shape, rank_one_gate_strides, 64),
                         .up = MutableTensorView(rank_one_up, DataType::Float32(), rank_one_up_shape,
                                                 rank_one_up_strides, 64)},
                        MakeIdentityPackedWeight(rank_one_packed, sizeof(rank_one_packed),
                                                 rank_one_packed_shape))
                        .ok());
    EXPECT_FLOAT_EQ(rank_one_gate[0], 0.5F);
    EXPECT_FLOAT_EQ(rank_one_gate[1], -1.5F);
    EXPECT_FLOAT_EQ(rank_one_up[0], -1.5F);
    EXPECT_FLOAT_EQ(rank_one_up[1], 2.0F);
    EXPECT_FLOAT_EQ(rank_one_up[2], -4.0F);

    constexpr int64_t input_shape[3] = {2, 2, 3};
    constexpr int64_t input_strides[3] = {12, 6, 1};
    constexpr int64_t gate_shape[3] = {2, 2, 2};
    constexpr int64_t gate_strides[3] = {12, 6, 1};
    constexpr int64_t up_shape[3] = {2, 2, 3};
    constexpr int64_t up_strides[3] = {16, 8, 1};
    constexpr std::array<int64_t, 2> packed_shape{5, 3};
    alignas(64) std::array<float, 24> input{};
    alignas(64) std::array<float, 15> packed{};
    alignas(64) std::array<float, 24> gate{};
    alignas(64) std::array<float, 32> up{};
    for (int64_t row = 0; row < 4; ++row) {
        input[row * 6] = static_cast<float>(row + 1);
        input[row * 6 + 1] = -static_cast<float>(row);
        input[row * 6 + 2] = 0.5F * static_cast<float>(row + 1);
    }
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }

    const auto kernel = PrepareGateUpKernel(2, 3);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunGateUpEntry(
                        *kernel,
                        {.input = TensorView(input.data(), DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .gate = MutableTensorView(gate.data(), DataType::Float32(), gate_shape,
                                                   gate_strides, 64),
                         .up = MutableTensorView(up.data(), DataType::Float32(), up_shape,
                                                 up_strides, 64)},
                        MakeIdentityPackedWeight(packed.data(), packed.size() * sizeof(float),
                                                 packed_shape))
                        .ok());

    const auto expect_projection = [&](const float* output,
                                       int64_t output_features,
                                       int64_t output_row_stride,
                                       int64_t weight_row_offset) {
        for (int64_t row = 0; row < 4; ++row) {
            for (int64_t feature = 0; feature < output_features; ++feature) {
                double expected = 0.0;
                for (int64_t in_feature = 0; in_feature < 3; ++in_feature) {
                    expected += static_cast<double>(input[row * 6 + in_feature]) *
                                static_cast<double>(packed[(weight_row_offset + feature) * 3 +
                                                           in_feature]);
                }
                EXPECT_FLOAT_EQ(output[row * output_row_stride + feature],
                                static_cast<float>(expected));
            }
        }
    };
    expect_projection(gate.data(), 2, 6, 0);
    expect_projection(up.data(), 3, 8, 2);
}

TEST(CPUKernelGateUpLinear, ZeroInnerDimensionWritesZerosWithoutInputOrWeightStorage) {
    constexpr int64_t input_shape[2] = {2, 0};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t gate_shape[2] = {2, 1};
    constexpr int64_t gate_strides[2] = {1, 1};
    constexpr int64_t up_shape[2] = {2, 2};
    constexpr int64_t up_strides[2] = {2, 1};
    constexpr std::array<int64_t, 2> packed_shape{3, 0};
    alignas(64) float gate[2] = {1.0F, 1.0F};
    alignas(64) float up[4] = {1.0F, 1.0F, 1.0F, 1.0F};

    const auto kernel = PrepareGateUpKernel(1, 2);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const GateUpTestViews valid_views{
            .input = TensorView(nullptr, DataType::Float32(), input_shape, input_strides, 64),
            .gate = MutableTensorView(gate, DataType::Float32(), gate_shape, gate_strides, 64),
            .up = MutableTensorView(up, DataType::Float32(), up_shape, up_strides, 64),
    };
    ASSERT_TRUE(RunGateUpEntry(*kernel, valid_views,
                               MakeIdentityPackedWeight(nullptr, 0, packed_shape))
                        .ok());
    for (const float value: gate) EXPECT_FLOAT_EQ(value, 0.0F);
    for (const float value: up) EXPECT_FLOAT_EQ(value, 0.0F);
}

TEST(CPUKernelGateUpLinear, ZeroRowsAndZeroOutputSplitsDoNotRequireStorage) {
    constexpr int64_t zero_rows_input_shape[2] = {0, 3};
    constexpr int64_t zero_rows_input_strides[2] = {3, 1};
    constexpr int64_t zero_rows_gate_shape[2] = {0, 2};
    constexpr int64_t zero_rows_gate_strides[2] = {2, 1};
    constexpr int64_t zero_rows_up_shape[2] = {0, 1};
    constexpr int64_t zero_rows_up_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> zero_rows_packed_shape{3, 3};
    const auto kernel = PrepareGateUpKernel(2, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_TRUE(RunGateUpEntry(
                        *kernel,
                        {.input = TensorView(nullptr, DataType::Float32(), zero_rows_input_shape,
                                             zero_rows_input_strides, 64),
                         .gate = MutableTensorView(nullptr, DataType::Float32(), zero_rows_gate_shape,
                                                   zero_rows_gate_strides, 64),
                         .up = MutableTensorView(nullptr, DataType::Float32(), zero_rows_up_shape,
                                                 zero_rows_up_strides, 64)},
                        MakeIdentityPackedWeight(nullptr, 9 * sizeof(float),
                                                 zero_rows_packed_shape))
                        .ok());

    constexpr int64_t input_shape[2] = {1, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t empty_shape[2] = {1, 0};
    constexpr int64_t empty_strides[2] = {1, 1};
    constexpr int64_t up_shape[2] = {1, 1};
    constexpr int64_t up_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> one_split_shape{1, 2};
    alignas(64) const float input[2] = {2.0F, -3.0F};
    alignas(64) const float packed[2] = {1.0F, 2.0F};
    alignas(64) float up[1]{};
    const auto only_up_kernel = PrepareGateUpKernel(0, 1);
    ASSERT_TRUE(only_up_kernel.ok()) << only_up_kernel.status().ToString();
    ASSERT_TRUE(RunGateUpEntry(
                        *only_up_kernel,
                        {.input = TensorView(input, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .gate = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                                   empty_strides, 64),
                         .up = MutableTensorView(up, DataType::Float32(), up_shape,
                                                 up_strides, 64)},
                        MakeIdentityPackedWeight(packed, sizeof(packed), one_split_shape))
                        .ok());
    EXPECT_FLOAT_EQ(up[0], -4.0F);

    alignas(64) const float only_gate_packed[2] = {-1.0F, 0.5F};
    alignas(64) float gate[1]{};
    const auto only_gate_kernel = PrepareGateUpKernel(1, 0);
    ASSERT_TRUE(only_gate_kernel.ok()) << only_gate_kernel.status().ToString();
    ASSERT_TRUE(RunGateUpEntry(
                        *only_gate_kernel,
                        {.input = TensorView(input, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .gate = MutableTensorView(gate, DataType::Float32(), up_shape,
                                                   up_strides, 64),
                         .up = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                                 empty_strides, 64)},
                        MakeIdentityPackedWeight(only_gate_packed, sizeof(only_gate_packed),
                                                 one_split_shape))
                        .ok());
    EXPECT_FLOAT_EQ(gate[0], -3.5F);

    constexpr std::array<int64_t, 2> empty_packed_shape{0, 2};
    const auto empty_kernel = PrepareGateUpKernel(0, 0);
    ASSERT_TRUE(empty_kernel.ok()) << empty_kernel.status().ToString();
    EXPECT_TRUE(RunGateUpEntry(
                        *empty_kernel,
                        {.input = TensorView(input, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .gate = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                                   empty_strides, 64),
                         .up = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                                 empty_strides, 64)},
                        MakeIdentityPackedWeight(nullptr, 0, empty_packed_shape))
                        .ok());
}

TEST(CPUKernelGateUpLinear, RejectsAliasingAndUndecidablePackedStorageOverlap) {
    constexpr int64_t input_shape[2] = {1, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t gate_shape[2] = {1, 2};
    constexpr int64_t gate_strides[2] = {2, 1};
    constexpr int64_t scalar_shape[2] = {1, 1};
    constexpr int64_t scalar_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{3, 3};
    alignas(64) float input[3] = {1.0F, 2.0F, 3.0F};
    alignas(64) float packed[9]{};
    alignas(64) float up[1]{};
    const auto kernel = PrepareGateUpKernel(2, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    EXPECT_EQ(RunGateUpEntry(
                      *kernel,
                      {.input = TensorView(input, DataType::Float32(), input_shape, input_strides, 64),
                       .gate = MutableTensorView(input, DataType::Float32(), gate_shape,
                                                 gate_strides, 64),
                       .up = MutableTensorView(up, DataType::Float32(), scalar_shape,
                                               scalar_strides, 64)},
                      MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
    alignas(64) float separate_input[3]{};
    alignas(64) float shared_output[2]{};
    EXPECT_EQ(RunGateUpEntry(
                      *kernel,
                      {.input = TensorView(separate_input, DataType::Float32(), input_shape,
                                           input_strides, 64),
                       .gate = MutableTensorView(shared_output, DataType::Float32(), gate_shape,
                                                 gate_strides, 64),
                       .up = MutableTensorView(shared_output, DataType::Float32(), scalar_shape,
                                               scalar_strides, 64)},
                      MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(RunGateUpEntry(
                      *kernel,
                      {.input = TensorView(separate_input, DataType::Float32(), input_shape,
                                           input_strides, 64),
                       .gate = MutableTensorView(packed, DataType::Float32(), gate_shape,
                                                 gate_strides, 64),
                       .up = MutableTensorView(up, DataType::Float32(), scalar_shape,
                                               scalar_strides, 64)},
                      MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr int64_t hole_input_shape[2] = {1, 1};
    constexpr int64_t hole_input_strides[2] = {1, 1};
    constexpr int64_t hole_gate_shape[2] = {1, 2};
    constexpr int64_t hole_gate_strides[2] = {32, 16};
    constexpr int64_t empty_shape[2] = {1, 0};
    constexpr int64_t empty_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> hole_packed_shape{2, 1};
    alignas(64) float hole_input[1] = {1.0F};
    alignas(64) float hole_storage[20]{};
    const auto hole_kernel = PrepareGateUpKernel(2, 0);
    ASSERT_TRUE(hole_kernel.ok()) << hole_kernel.status().ToString();
    EXPECT_EQ(RunGateUpEntry(
                      *hole_kernel,
                      {.input = TensorView(hole_input, DataType::Float32(), hole_input_shape,
                                           hole_input_strides, 64),
                       .gate = MutableTensorView(hole_storage, DataType::Float32(), hole_gate_shape,
                                                 hole_gate_strides, 64),
                       .up = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                               empty_strides, 64)},
                      MakeIdentityPackedWeight(hole_storage + 1, 2 * sizeof(float),
                                               hole_packed_shape))
                      .code(),
              StatusCode::kUnimplemented);
}

TEST(CPUKernelGateUpLinear, RejectsMalformedPackedMetadataAndTensorViews) {
    constexpr int64_t input_shape[2] = {1, 1};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t gate_shape[2] = {1, 1};
    constexpr int64_t gate_strides[2] = {1, 1};
    constexpr int64_t up_shape[2] = {1, 2};
    constexpr int64_t up_strides[2] = {2, 1};
    constexpr int64_t wrong_gate_shape[2] = {1, 2};
    constexpr std::array<int64_t, 2> packed_shape{3, 1};
    constexpr std::array<int64_t, 2> wrong_packed_shape{2, 1};
    alignas(64) float input[1]{};
    alignas(64) float packed[3]{};
    alignas(64) float gate[2]{};
    alignas(64) float up[2]{};
    const GateUpTestViews views{
            .input = TensorView(input, DataType::Float32(), input_shape, input_strides, 64),
            .gate = MutableTensorView(gate, DataType::Float32(), gate_shape, gate_strides, 64),
            .up = MutableTensorView(up, DataType::Float32(), up_shape, up_strides, 64),
    };
    const auto kernel = PrepareGateUpKernel(1, 2);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    EXPECT_EQ(RunGateUpEntry(*kernel, views,
                             MakeIdentityPackedWeight(packed, sizeof(packed), wrong_packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
    PackedWeightView wrong_recipe =
            MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape);
    wrong_recipe.logical_dtype = DataType::Float(16);
    EXPECT_EQ(RunGateUpEntry(*kernel, views, wrong_recipe).code(), StatusCode::kInvalidArgument);
    wrong_recipe = MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape);
    wrong_recipe.recipe_layout = "different_layout";
    EXPECT_EQ(RunGateUpEntry(*kernel, views, wrong_recipe).code(), StatusCode::kInvalidArgument);
    wrong_recipe = MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape);
    wrong_recipe.recipe_alignment = 32;
    EXPECT_EQ(RunGateUpEntry(*kernel, views, wrong_recipe).code(), StatusCode::kInvalidArgument);
    wrong_recipe = MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape);
    wrong_recipe.alignment = 32;
    EXPECT_EQ(RunGateUpEntry(*kernel, views, wrong_recipe).code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(RunGateUpEntry(*kernel, views,
                             MakeIdentityPackedWeight(packed, 2 * sizeof(float), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    GateUpTestViews wrong_dtype = views;
    wrong_dtype.input = TensorView(input, DataType::Float(16), input_shape, input_strides, 64);
    EXPECT_EQ(RunGateUpEntry(*kernel, wrong_dtype,
                             MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
    GateUpTestViews wrong_shape = views;
    wrong_shape.gate = MutableTensorView(gate, DataType::Float32(), wrong_gate_shape,
                                         gate_strides, 64);
    EXPECT_EQ(RunGateUpEntry(*kernel, wrong_shape,
                             MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    const std::array<TensorView, 1> inputs{views.input};
    const std::array<MutableTensorView, 2> outputs{views.gate, views.up};
    PreparedKernelParams prepared;
    EXPECT_EQ(kernel->params_builder(
                            KernelParamsBuildContext{
                                    .inputs = inputs,
                                    .outputs = outputs,
                                    .attrs = {},
                                    .packed_weight = MakeIdentityPackedWeight(
                                            packed, sizeof(packed), packed_shape),
                            },
                            prepared.storage.data())
                      .code(),
              StatusCode::kInvalidArgument);
    const cpu::detail::GateUpLinearF32KernelMetadata invalid_metadata{
            .gate_out_features = -1,
            .up_out_features = 2,
    };
    const auto invalid_metadata_attrs =
            std::as_bytes(std::span{&invalid_metadata, size_t{1}});
    EXPECT_EQ(kernel->params_builder(
                            KernelParamsBuildContext{
                                    .inputs = inputs,
                                    .outputs = outputs,
                                    .attrs = invalid_metadata_attrs,
                                    .packed_weight = MakeIdentityPackedWeight(
                                            packed, sizeof(packed), packed_shape),
                            },
                            prepared.storage.data())
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelGateUpLinear, PackedArtifactParticipatesInDeferredShapeChecks) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_features = ShapeSymbol::Create();
    const ShapeSymbol packed_features = ShapeSymbol::Create();
    const TensorSpec input_spec{.dtype = DataType::Float32(),
                                .shape = SymbolicShape({batch, input_features})};
    const TensorSpec weight_spec{.dtype = DataType::Float32(),
                                 .shape = SymbolicShape({ShapeSymbol::CreateFromValue(5),
                                                         packed_features})};
    const std::vector<TensorSpec> input_specs{input_spec, weight_spec};
    const GateUpLinearParams params{.gate_out_features = 2, .up_out_features = 3};
    const auto inferred = InferOperator(OpType::kGateUpLinear, OpParams{params}, input_specs);
    ASSERT_TRUE(inferred.ok()) << inferred.status().ToString();
    ASSERT_EQ(inferred->runtime_checks.size(), 1U);

    alignas(64) float logical_weight[15]{};
    constexpr int64_t logical_shape[2] = {5, 3};
    constexpr int64_t logical_strides[2] = {3, 1};
    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kGateUpLinear,
            TensorView(logical_weight, DataType::Float32(), logical_shape, logical_strides, 64),
            MakeGateUpSelector());
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    PackedWeightStore store;
    const WeightArtifactKey key{
            .source_id = 0,
            .value_index = 1,
            .binding = {},
            .selector = MakeGateUpSelector(),
            .recipe = CpuWeightPrepacker::RecipeFor(MakeGateUpSelector()),
    };
    ASSERT_TRUE(store.Store(key, std::shared_ptr<const PackedWeights>(std::move(*packed))).ok());

    RuntimeBuilder runtime_builder;
    runtime_builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    Runtime runtime = runtime_builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(
            runtime, store,
            std::vector{ExecutionPlanNodeSpec{
                    .op_type = OpType::kGateUpLinear,
                    .selector = MakeGateUpSelector(),
                    .input_specs = input_specs,
                    .output_specs = inferred->outputs,
                    .runtime_checks = inferred->runtime_checks,
                    .op_params = OpParams{params},
            }});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->steps().front().kernel_input_ports, std::vector<uint32_t>({0}));

    alignas(64) float input[4]{};
    constexpr int64_t input_shape[2] = {1, 4};
    constexpr int64_t input_strides[2] = {4, 1};
    const auto bindings = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = plan->steps().front().inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides, 64)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

class FloatRawStorage final : public RawStorage {
public:
    std::vector<float> values{};
};

RawWeightView MakeRawWeight(const std::shared_ptr<FloatRawStorage>& storage,
                            size_t offset,
                            std::initializer_list<int64_t> shape) {
    size_t elements = 1;
    for (const int64_t dim: shape) {
        elements *= static_cast<size_t>(dim);
    }
    return RawWeightView{
            .data = reinterpret_cast<const std::byte*>(storage->values.data() + offset),
            .bytes = elements * sizeof(float),
            .dtype = DataType::Float32(),
            .shape = std::vector<int64_t>(shape),
            .storage = storage,
            .is_contiguous = true,
    };
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return SymbolicShape(IntArrayView{copied});
}

StatusOr<GraphValueId> AddProjection(ModelGraph& graph,
                                     GraphValueId input,
                                     int64_t output_features,
                                     TransformerWeightRole role) {
    const GraphValueId weight = graph.AddWeight(
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = StaticShape({output_features, 3})},
            MakeTransformerWeightBinding(0U, role));
    auto node = graph.AddNode(
            OpType::kLinear, 0U, {input, weight},
            {NodeOutputDesc{.payload = ActivationValue{}}}, LinearParams{});
    if (!node.ok()) {
        return node.status();
    }
    return node->outputs[0];
}

TEST(CPUKernelGateUpLinear, FusedPackedExecutionRequiresNoExternalGateUpWeightBinding) {
    ModelGraph graph;
    const TensorSpec activation_spec{.dtype = DataType::Float32(),
                                     .shape = StaticShape({2, 3})};
    const GraphValueId lhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const GraphValueId rhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const auto bridge = graph.AddNode(
            OpType::kAdd, std::nullopt, {lhs, rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    ASSERT_TRUE(bridge.ok()) << bridge.status().ToString();
    const auto gate = AddProjection(graph, bridge->outputs[0], 2,
                                    TransformerWeightRole::kMlpGate);
    const auto up = AddProjection(graph, bridge->outputs[0], 3,
                                  TransformerWeightRole::kMlpUp);
    ASSERT_TRUE(gate.ok()) << gate.status().ToString();
    ASSERT_TRUE(up.ok()) << up.status().ToString();
    graph.MarkOutput(*gate);
    graph.MarkOutput(*up);

    PassContext optimization;
    optimization.const_eval_policy.max_output_bytes = 0;
    const auto optimized = OptimizeModelGraph(graph, optimization);
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    ASSERT_EQ(optimized->FindNodesByOpType(OpType::kGateUpLinear).size(), 1U);
    const auto lowered = LowerModelGraph(
            *optimized, GraphLoweringConfig{.enable_packed_weights = true});
    ASSERT_TRUE(lowered.ok()) << lowered.status().ToString();

    auto raw = std::make_shared<FloatRawStorage>();
    raw->values = {
            1.0F,
            0.0F,
            -1.0F,
            0.5F,
            1.0F,
            0.0F,
            -1.0F,
            0.5F,
            1.0F,
            2.0F,
            0.0F,
            0.0F,
            0.0F,
            2.0F,
            0.0F,
    };
    ResolvedModelWeights resolved;
    resolved.layers.resize(1);
    resolved.layers[0].mlp.gate_proj = MakeRawWeight(raw, 0, {2, 3});
    resolved.layers[0].mlp.up_proj = MakeRawWeight(raw, 6, {3, 3});
    auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    ASSERT_EQ(requests->size(), 1U);
    EXPECT_EQ(requests->front().op_type, OpType::kGateUpLinear);
    ASSERT_EQ(requests->front().components.size(), 2U);

    PackedWeightStore packed_store;
    RuntimeBuilder runtime_builder;
    runtime_builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    Runtime runtime = runtime_builder.Build();
    const auto prepack_backend = runtime.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(prepack_backend.ok()) << prepack_backend.status().ToString();
    for (WeightPackingRequest& request: *requests) {
        const auto recipe = (*prepack_backend)->GetPackingRecipe(request.op_type, request.selector);
        ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
        request.recipe = *recipe;
    }
    ASSERT_TRUE(PrepackWeightRequests(**prepack_backend, packed_store, *requests).ok());
    const auto plan = ExecutionPlanBuilder::Build(runtime, packed_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    const ExecutionStep* gate_up_step = nullptr;
    const ExecutionStep* add_step = nullptr;
    for (const ExecutionStep& step: plan->steps()) {
        if (step.kernel.op_type == OpType::kGateUpLinear) {
            gate_up_step = &step;
        } else if (step.kernel.op_type == OpType::kAdd) {
            add_step = &step;
        }
    }
    ASSERT_NE(gate_up_step, nullptr);
    ASSERT_NE(add_step, nullptr);
    EXPECT_EQ(gate_up_step->kernel_input_ports, std::vector<uint32_t>({0}));
    ASSERT_NE(gate_up_step->packed_weights, nullptr);

    alignas(64) const float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F};
    alignas(64) float gate_output[4]{};
    alignas(64) float up_output[6]{};
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t gate_shape[2] = {2, 2};
    constexpr int64_t gate_strides[2] = {2, 1};
    constexpr int64_t up_shape[2] = {2, 3};
    constexpr int64_t up_strides[2] = {3, 1};
    auto prepared = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = add_step->inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides, 64)},
                          {.value = add_step->inputs[1],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides, 64)}},
             .writable = {{.value = gate_up_step->outputs[0],
                           .tensor = MutableTensorView(gate_output, DataType::Float32(), gate_shape,
                                                       gate_strides, 64)},
                          {.value = gate_up_step->outputs[1],
                           .tensor = MutableTensorView(up_output, DataType::Float32(), up_shape,
                                                       up_strides, 64)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    ASSERT_TRUE(Executor::Execute(*plan, *context).ok());

    const auto expect_projection = [&](const float* output,
                                       int64_t output_features,
                                       int64_t weight_offset) {
        for (int64_t row = 0; row < 2; ++row) {
            for (int64_t feature = 0; feature < output_features; ++feature) {
                double expected = 0.0;
                for (int64_t in_feature = 0; in_feature < 3; ++in_feature) {
                    expected += static_cast<double>(2.0F * input[row * 3 + in_feature]) *
                                static_cast<double>(raw->values[(weight_offset + feature) * 3 +
                                                                in_feature]);
                }
                EXPECT_FLOAT_EQ(output[row * output_features + feature],
                                static_cast<float>(expected));
            }
        }
    };
    expect_projection(gate_output, 2, 0);
    expect_projection(up_output, 3, 2);
}

} // namespace
