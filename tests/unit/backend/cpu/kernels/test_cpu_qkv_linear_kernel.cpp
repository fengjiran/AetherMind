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
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/model/weight_prepack_planner.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/qkv_linear/qkv_linear_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;

KernelSelector MakeQkvSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareQkvKernel(int64_t q_features,
                                          int64_t k_features,
                                          int64_t v_features,
                                          bool has_bias = false) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kQkvLinear, MakeQkvSelector(),
            OpParams{QkvLinearParams{.q_out_features = q_features,
                                     .k_out_features = k_features,
                                     .v_out_features = v_features,
                                     .has_bias = has_bias}});
}

struct QkvTestViews {
    TensorView input{};
    MutableTensorView query{};
    MutableTensorView key{};
    MutableTensorView value{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunQkvEntry(const ResolvedKernel& kernel,
                   const QkvTestViews& views,
                   PackedWeightBuildView packed) {
    const std::array<TensorView, 1> inputs{views.input};
    const std::array<MutableTensorView, 3> outputs{views.query, views.key, views.value};
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

PackedWeightBuildView MakeIdentityPackedWeight(const float* data,
                                               size_t nbytes,
                                               std::span<const int64_t> shape) {
    return PackedWeightBuildView{
            .data = data,
            .nbytes = nbytes,
            .logical_dtype = DataType::Float32(),
            .logical_shape = shape,
            .recipe_layout = "cpu_identity",
            .recipe_alignment = 64,
            .alignment = 64,
    };
}

TEST(CPUKernelQkvLinear, CpuBackendPreparesPackedF32ReferenceKernel) {
    const auto kernel = PrepareQkvKernel(4, 2, 2);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(kernel->op_type, OpType::kQkvLinear);
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::qkv_linear_f32_reference");
    EXPECT_NE(kernel->fn, nullptr);
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::QkvLinearF32KernelArgs));

    CpuBackend backend;
    KernelSelector plain = MakeQkvSelector();
    plain.weight_format = WeightFormat::kPlain;
    EXPECT_EQ(backend.PrepareKernel(
                             OpType::kQkvLinear, plain,
                             OpParams{QkvLinearParams{.q_out_features = 4,
                                                      .k_out_features = 2,
                                                      .v_out_features = 2}})
                      .status()
                      .code(),
              StatusCode::kNotFound);
    EXPECT_FALSE(PrepareQkvKernel(4, 2, 2, true).ok());
    EXPECT_EQ(PrepareQkvKernel(std::numeric_limits<int64_t>::max(), 1, 0)
                      .status()
                      .code(),
              StatusCode::kOverflow);
}

TEST(CPUKernelQkvLinear, ReferenceSupportsGqaAndStridedViews) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {5, 1};
    constexpr int64_t query_shape[2] = {2, 4};
    constexpr int64_t query_strides[2] = {6, 1};
    constexpr int64_t key_shape[2] = {2, 2};
    constexpr int64_t key_strides[2] = {4, 1};
    constexpr int64_t value_shape[2] = {2, 3};
    constexpr int64_t value_strides[2] = {5, 1};
    constexpr std::array<int64_t, 2> packed_shape{9, 3};
    alignas(64) std::array<float, 10> input{};
    alignas(64) std::array<float, 27> packed{};
    alignas(64) std::array<float, 12> query{};
    alignas(64) std::array<float, 8> key{};
    alignas(64) std::array<float, 10> value{};

    input[0] = 1.0F;
    input[1] = -2.0F;
    input[2] = 0.5F;
    input[5] = -1.0F;
    input[6] = 3.0F;
    input[7] = 2.0F;
    for (size_t i = 0; i < packed.size(); ++i) {
        packed[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25F;
    }

    const auto kernel = PrepareQkvKernel(4, 2, 3);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunQkvEntry(
                        *kernel,
                        {.input = TensorView(input.data(), DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .query = MutableTensorView(query.data(), DataType::Float32(), query_shape,
                                                    query_strides, 64),
                         .key = MutableTensorView(key.data(), DataType::Float32(), key_shape,
                                                  key_strides, 64),
                         .value = MutableTensorView(value.data(), DataType::Float32(), value_shape,
                                                    value_strides, 64)},
                        MakeIdentityPackedWeight(packed.data(), packed.size() * sizeof(float),
                                                 packed_shape))
                        .ok());

    const auto expect_projection = [&](const float* output,
                                       int64_t output_features,
                                       int64_t output_row_stride,
                                       int64_t weight_row_offset) {
        for (int64_t row = 0; row < 2; ++row) {
            for (int64_t feature = 0; feature < output_features; ++feature) {
                double expected = 0.0;
                for (int64_t in_feature = 0; in_feature < 3; ++in_feature) {
                    expected += static_cast<double>(input[row * input_strides[0] + in_feature]) *
                                static_cast<double>(packed[(weight_row_offset + feature) * 3 + in_feature]);
                }
                EXPECT_FLOAT_EQ(output[row * output_row_stride + feature],
                                static_cast<float>(expected));
            }
        }
    };
    expect_projection(query.data(), 4, query_strides[0], 0);
    expect_projection(key.data(), 2, key_strides[0], 4);
    expect_projection(value.data(), 3, value_strides[0], 6);
}

TEST(CPUKernelQkvLinear, ZeroInnerDimensionWritesZeroWithoutInputOrWeightStorage) {
    constexpr int64_t input_shape[2] = {2, 0};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t output_shape[2] = {2, 1};
    constexpr int64_t output_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{3, 0};
    alignas(64) float query[2] = {1.0F, 1.0F};
    alignas(64) float key[2] = {1.0F, 1.0F};
    alignas(64) float value[2] = {1.0F, 1.0F};

    const auto kernel = PrepareQkvKernel(1, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunQkvEntry(
                        *kernel,
                        {.input = TensorView(nullptr, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .query = MutableTensorView(query, DataType::Float32(), output_shape,
                                                    output_strides, 64),
                         .key = MutableTensorView(key, DataType::Float32(), output_shape,
                                                  output_strides, 64),
                         .value = MutableTensorView(value, DataType::Float32(), output_shape,
                                                    output_strides, 64)},
                        MakeIdentityPackedWeight(nullptr, 0, packed_shape))
                        .ok());
    EXPECT_FLOAT_EQ(query[0], 0.0F);
    EXPECT_FLOAT_EQ(query[1], 0.0F);
    EXPECT_FLOAT_EQ(key[0], 0.0F);
    EXPECT_FLOAT_EQ(key[1], 0.0F);
    EXPECT_FLOAT_EQ(value[0], 0.0F);
    EXPECT_FLOAT_EQ(value[1], 0.0F);
}

TEST(CPUKernelQkvLinear, ZeroRowCountDoesNotRequireTensorStorage) {
    constexpr int64_t input_shape[2] = {0, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t query_shape[2] = {0, 2};
    constexpr int64_t query_strides[2] = {2, 1};
    constexpr int64_t key_value_shape[2] = {0, 1};
    constexpr int64_t key_value_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{4, 3};

    const auto kernel = PrepareQkvKernel(2, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_TRUE(RunQkvEntry(
                        *kernel,
                        {.input = TensorView(nullptr, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .query = MutableTensorView(nullptr, DataType::Float32(), query_shape,
                                                    query_strides, 64),
                         .key = MutableTensorView(nullptr, DataType::Float32(), key_value_shape,
                                                  key_value_strides, 64),
                         .value = MutableTensorView(nullptr, DataType::Float32(), key_value_shape,
                                                    key_value_strides, 64)},
                        MakeIdentityPackedWeight(nullptr, 4 * 3 * sizeof(float), packed_shape))
                        .ok());
}

TEST(CPUKernelQkvLinear, ZeroQueryFeaturesLeaveOnlyKeyAndValueProjections) {
    constexpr int64_t input_shape[2] = {1, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t empty_shape[2] = {1, 0};
    constexpr int64_t empty_strides[2] = {1, 1};
    constexpr int64_t output_shape[2] = {1, 1};
    constexpr int64_t output_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{2, 2};
    alignas(64) const float input[2] = {2.0F, -3.0F};
    alignas(64) const float packed[4] = {1.0F, 2.0F, -1.0F, 0.5F};
    alignas(64) float key[1]{};
    alignas(64) float value[1]{};

    const auto kernel = PrepareQkvKernel(0, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    ASSERT_TRUE(RunQkvEntry(
                        *kernel,
                        {.input = TensorView(input, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .query = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                                    empty_strides, 64),
                         .key = MutableTensorView(key, DataType::Float32(), output_shape,
                                                  output_strides, 64),
                         .value = MutableTensorView(value, DataType::Float32(), output_shape,
                                                    output_strides, 64)},
                        MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                        .ok());
    EXPECT_FLOAT_EQ(key[0], -4.0F);
    EXPECT_FLOAT_EQ(value[0], -3.5F);
}

TEST(CPUKernelQkvLinear, AllZeroOutputFeaturesDoNotRequirePackedStorage) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 0};
    constexpr int64_t output_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{0, 3};
    alignas(64) float input[6]{};

    const auto kernel = PrepareQkvKernel(0, 0, 0);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_TRUE(RunQkvEntry(
                        *kernel,
                        {.input = TensorView(input, DataType::Float32(), input_shape,
                                             input_strides, 64),
                         .query = MutableTensorView(nullptr, DataType::Float32(), output_shape,
                                                    output_strides, 64),
                         .key = MutableTensorView(nullptr, DataType::Float32(), output_shape,
                                                  output_strides, 64),
                         .value = MutableTensorView(nullptr, DataType::Float32(), output_shape,
                                                    output_strides, 64)},
                        MakeIdentityPackedWeight(nullptr, 0, packed_shape))
                        .ok());
}

TEST(CPUKernelQkvLinear, RejectsInputOutputAndPackedWeightAliasing) {
    constexpr int64_t input_shape[2] = {1, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t query_shape[2] = {1, 2};
    constexpr int64_t query_strides[2] = {2, 1};
    constexpr int64_t scalar_shape[2] = {1, 1};
    constexpr int64_t scalar_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{4, 3};
    alignas(64) float input[3] = {1.0F, 2.0F, 3.0F};
    alignas(64) float packed[12]{};
    alignas(64) float key[1]{};
    alignas(64) float value[1]{};
    const auto kernel = PrepareQkvKernel(2, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const QkvTestViews input_alias{
            .input = TensorView(input, DataType::Float32(), input_shape, input_strides, 64),
            .query = MutableTensorView(input, DataType::Float32(), query_shape, query_strides, 64),
            .key = MutableTensorView(key, DataType::Float32(), scalar_shape, scalar_strides, 64),
            .value = MutableTensorView(value, DataType::Float32(), scalar_shape, scalar_strides, 64),
    };
    EXPECT_EQ(RunQkvEntry(*kernel, input_alias,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    alignas(64) float separate_input[3] = {1.0F, 2.0F, 3.0F};
    const QkvTestViews packed_alias{
            .input = TensorView(separate_input, DataType::Float32(), input_shape, input_strides, 64),
            .query = MutableTensorView(packed, DataType::Float32(), query_shape, query_strides, 64),
            .key = MutableTensorView(key, DataType::Float32(), scalar_shape, scalar_strides, 64),
            .value = MutableTensorView(value, DataType::Float32(), scalar_shape, scalar_strides, 64),
    };
    EXPECT_EQ(RunQkvEntry(*kernel, packed_alias,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    alignas(64) float distinct_input[3] = {1.0F, 2.0F, 3.0F};
    alignas(64) float shared_output[2]{};
    const QkvTestViews output_alias{
            .input = TensorView(distinct_input, DataType::Float32(), input_shape, input_strides, 64),
            .query = MutableTensorView(shared_output, DataType::Float32(), query_shape, query_strides, 64),
            .key = MutableTensorView(shared_output, DataType::Float32(), scalar_shape, scalar_strides, 64),
            .value = MutableTensorView(value, DataType::Float32(), scalar_shape, scalar_strides, 64),
    };
    EXPECT_EQ(RunQkvEntry(*kernel, output_alias,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelQkvLinear, RejectsMalformedPackedMetadataAndTensorViews) {
    constexpr int64_t input_shape[2] = {1, 1};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t output_shape[2] = {1, 1};
    constexpr int64_t output_strides[2] = {1, 1};
    constexpr int64_t wrong_output_shape[2] = {1, 2};
    constexpr std::array<int64_t, 2> packed_shape{3, 1};
    constexpr std::array<int64_t, 2> wrong_packed_shape{2, 1};
    alignas(64) float input[1]{};
    alignas(64) float packed[3]{};
    alignas(64) float query[2]{};
    alignas(64) float key[1]{};
    alignas(64) float value[1]{};
    const QkvTestViews views{
            .input = TensorView(input, DataType::Float32(), input_shape, input_strides, 64),
            .query = MutableTensorView(query, DataType::Float32(), output_shape, output_strides, 64),
            .key = MutableTensorView(key, DataType::Float32(), output_shape, output_strides, 64),
            .value = MutableTensorView(value, DataType::Float32(), output_shape, output_strides, 64),
    };
    const auto kernel = PrepareQkvKernel(1, 1, 1);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    EXPECT_EQ(RunQkvEntry(*kernel, views,
                          MakeIdentityPackedWeight(packed, sizeof(packed), wrong_packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    PackedWeightBuildView wrong_recipe =
            MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape);
    wrong_recipe.recipe_layout = "different_layout";
    EXPECT_EQ(RunQkvEntry(*kernel, views, wrong_recipe).code(), StatusCode::kInvalidArgument);

    EXPECT_EQ(RunQkvEntry(*kernel, views,
                          MakeIdentityPackedWeight(packed, 2 * sizeof(float), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    QkvTestViews wrong_dtype = views;
    wrong_dtype.input = TensorView(input, DataType::Float(16), input_shape, input_strides, 64);
    EXPECT_EQ(RunQkvEntry(*kernel, wrong_dtype,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    QkvTestViews wrong_shape = views;
    wrong_shape.query = MutableTensorView(query, DataType::Float32(), wrong_output_shape,
                                          output_strides, 64);
    EXPECT_EQ(RunQkvEntry(*kernel, wrong_shape,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);

    constexpr std::array<int64_t, 0> scalar_shape{};
    constexpr std::array<int64_t, 0> scalar_strides{};
    QkvTestViews rank_zero = views;
    rank_zero.input = TensorView(input, DataType::Float32(), scalar_shape, scalar_strides, 64);
    EXPECT_EQ(RunQkvEntry(*kernel, rank_zero,
                          MakeIdentityPackedWeight(packed, sizeof(packed), packed_shape))
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelQkvLinear, ReportsUnimplementedForPackedStorageOnlyInOutputColumnHole) {
    constexpr int64_t input_shape[2] = {1, 1};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t query_shape[2] = {1, 2};
    constexpr int64_t query_strides[2] = {32, 16};
    constexpr int64_t empty_shape[2] = {1, 0};
    constexpr int64_t empty_strides[2] = {1, 1};
    constexpr std::array<int64_t, 2> packed_shape{2, 1};
    alignas(64) float input[1] = {1.0F};
    alignas(64) float output_storage[20]{};

    const auto kernel = PrepareQkvKernel(2, 0, 0);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    const Status status = RunQkvEntry(
            *kernel,
            {.input = TensorView(input, DataType::Float32(), input_shape, input_strides, 64),
             .query = MutableTensorView(output_storage, DataType::Float32(), query_shape,
                                        query_strides, 64),
             .key = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                      empty_strides, 64),
             .value = MutableTensorView(nullptr, DataType::Float32(), empty_shape,
                                        empty_strides, 64)},
            MakeIdentityPackedWeight(output_storage + 1, 2 * sizeof(float), packed_shape));

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
}

TEST(CPUKernelQkvLinear, PackedArtifactParticipatesInDeferredShapeChecks) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_features = ShapeSymbol::Create();
    const ShapeSymbol packed_features = ShapeSymbol::Create();
    const TensorSpec input_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape({batch, input_features}),
    };
    const TensorSpec weight_spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape({ShapeSymbol::CreateFromValue(4), packed_features}),
    };
    const std::vector<TensorSpec> input_specs{input_spec, weight_spec};
    const QkvLinearParams params{
            .q_out_features = 2,
            .k_out_features = 1,
            .v_out_features = 1,
    };
    const auto inferred = InferOperator(OpType::kQkvLinear, OpParams{params}, input_specs);
    ASSERT_TRUE(inferred.ok()) << inferred.status().ToString();
    ASSERT_EQ(inferred->runtime_checks.size(), 1U);

    alignas(64) float logical_weight[12]{};
    constexpr int64_t logical_shape[2] = {4, 3};
    constexpr int64_t logical_strides[2] = {3, 1};
    CpuWeightPrepacker prepacker;
    auto packed = prepacker.Pack(
            OpType::kQkvLinear,
            TensorView(logical_weight, DataType::Float32(), logical_shape,
                       logical_strides, 64),
            MakeQkvSelector());
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    PackedWeightStore store;
    const WeightArtifactKey key{
            .source_id = 0,
            .value_index = 1,
            .binding = {},
            .selector = MakeQkvSelector(),
            .recipe = CpuWeightPrepacker::RecipeFor(MakeQkvSelector()),
    };
    ASSERT_TRUE(store.Store(key, std::shared_ptr<const PackedWeights>(std::move(*packed))).ok());

    RuntimeBuilder runtime_builder;
    runtime_builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    Runtime runtime = runtime_builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(
            runtime, store,
            std::vector{ExecutionPlanNodeSpec{
                    .op_type = OpType::kQkvLinear,
                    .selector = MakeQkvSelector(),
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

TEST(CPUKernelQkvLinear, FusedPackedExecutionRequiresNoExternalQkvWeightBinding) {
    ModelGraph graph;
    const TensorSpec activation_spec{.dtype = DataType::Float32(),
                                     .shape = StaticShape({2, 3})};
    const GraphValueId lhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const GraphValueId rhs = graph.AddConstant(activation_spec, ConstantBinding{});
    const auto bridge = graph.AddNode(
            OpType::kAdd, std::nullopt, {lhs, rhs},
            {NodeOutputDesc{.payload = ActivationValue{}}}, AddParams{});
    ASSERT_TRUE(bridge.ok()) << bridge.status().ToString();
    const auto query = AddProjection(graph, bridge->outputs[0], 4,
                                     TransformerWeightRole::kAttentionQ);
    const auto key = AddProjection(graph, bridge->outputs[0], 2,
                                   TransformerWeightRole::kAttentionK);
    const auto value = AddProjection(graph, bridge->outputs[0], 2,
                                     TransformerWeightRole::kAttentionV);
    ASSERT_TRUE(query.ok()) << query.status().ToString();
    ASSERT_TRUE(key.ok()) << key.status().ToString();
    ASSERT_TRUE(value.ok()) << value.status().ToString();
    graph.MarkOutput(*query);
    graph.MarkOutput(*key);
    graph.MarkOutput(*value);

    PassContext optimization;
    optimization.const_eval_policy.max_output_bytes = 0;
    const auto optimized = OptimizeModelGraph(graph, optimization);
    ASSERT_TRUE(optimized.ok()) << optimized.status().ToString();
    ASSERT_EQ(optimized->FindNodesByOpType(OpType::kQkvLinear).size(), 1U);
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
            -0.5F,
            0.0F,
            1.0F,
            1.0F,
            1.0F,
            1.0F,
            2.0F,
            0.0F,
            0.0F,
            0.0F,
            2.0F,
            0.0F,
            0.0F,
            0.0F,
            2.0F,
            1.0F,
            -1.0F,
            0.5F,
    };
    ResolvedModelWeights resolved;
    resolved.layers.resize(1);
    resolved.layers[0].attn.q_proj = MakeRawWeight(raw, 0, {4, 3});
    resolved.layers[0].attn.k_proj = MakeRawWeight(raw, 12, {2, 3});
    resolved.layers[0].attn.v_proj = MakeRawWeight(raw, 18, {2, 3});
    const auto requests = BuildWeightPackingRequests(*lowered, resolved);
    ASSERT_TRUE(requests.ok()) << requests.status().ToString();
    ASSERT_EQ(requests->size(), 1U);
    EXPECT_EQ(requests->front().op_type, OpType::kQkvLinear);

    PackedWeightStore packed_store;
    ASSERT_TRUE(WeightPrepackPlanner::PrepackAndStore(packed_store, *requests).ok());
    RuntimeBuilder runtime_builder;
    runtime_builder.RegisterBackendFactory(
            DeviceType::kCPU, std::make_unique<CpuBackendFactory>());
    Runtime runtime = runtime_builder.Build();
    const auto plan = ExecutionPlanBuilder::Build(runtime, packed_store, *lowered);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    const ExecutionStep* qkv_step = nullptr;
    const ExecutionStep* add_step = nullptr;
    for (const ExecutionStep& step: plan->steps()) {
        if (step.kernel.op_type == OpType::kQkvLinear) {
            qkv_step = &step;
        } else if (step.kernel.op_type == OpType::kAdd) {
            add_step = &step;
        }
    }
    ASSERT_NE(qkv_step, nullptr);
    ASSERT_NE(add_step, nullptr);
    EXPECT_EQ(qkv_step->kernel_input_ports, std::vector<uint32_t>({0}));
    ASSERT_NE(qkv_step->packed_weights, nullptr);

    alignas(64) const float input[6] = {1.0F, 2.0F, 3.0F, -1.0F, 0.5F, 2.0F};
    alignas(64) float query_output[8]{};
    alignas(64) float key_output[4]{};
    alignas(64) float value_output[4]{};
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t query_shape[2] = {2, 4};
    constexpr int64_t query_strides[2] = {4, 1};
    constexpr int64_t key_value_shape[2] = {2, 2};
    constexpr int64_t key_value_strides[2] = {2, 1};
    auto prepared = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = add_step->inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides, 64)},
                          {.value = add_step->inputs[1],
                           .tensor = TensorView(input, DataType::Float32(), input_shape,
                                                input_strides, 64)}},
             .writable = {{.value = qkv_step->outputs[0],
                           .tensor = MutableTensorView(query_output, DataType::Float32(), query_shape,
                                                       query_strides, 64)},
                          {.value = qkv_step->outputs[1],
                           .tensor = MutableTensorView(key_output, DataType::Float32(), key_value_shape,
                                                       key_value_strides, 64)},
                          {.value = qkv_step->outputs[2],
                           .tensor = MutableTensorView(value_output, DataType::Float32(), key_value_shape,
                                                       key_value_strides, 64)}}},
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
                                static_cast<double>(raw->values[(weight_offset + feature) * 3 + in_feature]);
                }
                EXPECT_FLOAT_EQ(output[row * output_features + feature],
                                static_cast<float>(expected));
            }
        }
    };
    expect_projection(query_output, 4, 0);
    expect_projection(key_output, 2, 4);
    expect_projection(value_output, 2, 6);
}

} // namespace
