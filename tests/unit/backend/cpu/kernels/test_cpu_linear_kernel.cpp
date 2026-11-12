#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"
#include "backend/cpu/kernels/linear/linear_internal.h"
#include "execution/test_execution_binding_helpers.h"
#include "test_utils/test_allocator.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
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

StatusOr<ResolvedKernel> PrepareLinearKernel(const CpuFeaturePolicy& policy) {
    CpuBackend backend(policy);
    return backend.PrepareKernel(OpType::kLinear, MakeLinearSelector(), OpParams{LinearParams{}});
}

StatusOr<ResolvedKernel> PrepareLinearW16Kernel(DataType weight_dtype,
                                                ExecPhase phase = ExecPhase::kBoth) {
    CpuBackend backend;
    KernelSelector selector = MakeLinearSelector();
    selector.weight_dtype = weight_dtype;
    selector.phase = phase;
    return backend.PrepareKernel(OpType::kLinear, selector, OpParams{LinearParams{}});
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

uint16_t ReducedBits(float value, DataType dtype) {
    if (dtype == DataType::Float(16)) {
        return Half(value).bits();
    }
    return BFloat16(value).x;
}

float WidenReducedBits(uint16_t bits, DataType dtype) {
    const int fraction_bits = dtype == DataType::Float(16) ? 10 : 7;
    const int bias = fraction_bits == 10 ? 15 : 127;
    const unsigned exponent_mask = fraction_bits == 10 ? 31U : 255U;
    const unsigned exponent = (bits >> fraction_bits) & exponent_mask;
    const unsigned fraction = bits & ((1U << fraction_bits) - 1);
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16;
    if (exponent == exponent_mask) {
        return std::bit_cast<float>(sign | 0x7f800000U |
                                    (fraction == 0 ? 0U : (fraction << (23 - fraction_bits)) | 0x00400000U));
    }
    const double significand = exponent == 0 ? fraction : (1U << fraction_bits) + fraction;
    const float value = static_cast<float>(std::ldexp(significand,
                                                      static_cast<int>(exponent == 0 ? 1 : exponent) - bias - fraction_bits));
    return std::bit_cast<float>(sign | std::bit_cast<uint32_t>(value));
}

void StoreReduced(std::byte* storage, int64_t index, float value, DataType dtype) {
    const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
    std::memcpy(storage + static_cast<size_t>(index) * sizeof(bits), &bits, sizeof(bits));
}

float FixedFp32Dot(const float* input,
                   int64_t input_stride,
                   const std::byte* weight,
                   int64_t weight_stride,
                   int64_t count,
                   DataType dtype) {
    float sum = 0.0F;
    for (int64_t index = 0; index < count; ++index) {
        uint16_t bits = 0;
        std::memcpy(&bits, weight + static_cast<size_t>(index * weight_stride) * sizeof(bits),
                    sizeof(bits));
        sum += input[index * input_stride] * WidenReducedBits(bits, dtype);
    }
    return sum;
}

std::pair<double, double> DoubleDotAndAbsoluteProductSum(
        const float* input,
        int64_t input_stride,
        const std::byte* weight,
        int64_t weight_stride,
        int64_t count,
        DataType dtype) {
    double sum = 0.0;
    double absolute_products = 0.0;
    for (int64_t index = 0; index < count; ++index) {
        uint16_t bits = 0;
        std::memcpy(&bits, weight + static_cast<size_t>(index * weight_stride) * sizeof(bits),
                    sizeof(bits));
        const double product = static_cast<double>(input[index * input_stride]) *
                               static_cast<double>(WidenReducedBits(bits, dtype));
        sum += product;
        absolute_products += std::abs(product);
    }
    return {sum, absolute_products};
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
    EXPECT_EQ(fp16.status(), StatusCode::kNotFound);
}

TEST(CPUKernelLinearW16, CpuBackendPreparesBothReducedWeightReferencesForBothPhases) {
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        for (const ExecPhase phase: {ExecPhase::kPrefill, ExecPhase::kDecode}) {
            const auto kernel = PrepareLinearW16Kernel(dtype, phase);
            ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
            EXPECT_EQ(kernel->params_size, sizeof(cpu::detail::LinearF32W16KernelArgs));
            EXPECT_EQ(std::string_view{kernel->name},
                      dtype == DataType::Float(16) ? "cpu::linear_f32_f16_reference"
                                                   : "cpu::linear_f32_bf16_reference");
        }
    }
}

TEST(CPUKernelLinearW16, FixedFp32AccumulationHandlesRankThreeStridesUnalignedWeightsAndTails) {
    constexpr int64_t input_shape[3] = {2, 1, 9};
    constexpr int64_t input_strides[3] = {13, 13, 1};
    constexpr int64_t weight_shape[2] = {3, 9};
    constexpr int64_t weight_strides[2] = {12, 1};
    constexpr int64_t output_shape[3] = {2, 1, 3};
    constexpr int64_t output_strides[3] = {7, 7, 1};

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        std::array<float, 22> input{};
        std::array<float, 13> output{};
        std::array<std::byte, 1 + 2 * 33> weight_storage{};
        std::byte* const weight = weight_storage.data() + 1;
        for (int64_t row = 0; row < 2; ++row) {
            for (int64_t k = 0; k < 9; ++k) {
                input[static_cast<size_t>(row * 13 + k)] =
                        static_cast<float>(row * 9 + k - 5) * 0.125F;
            }
        }
        for (int64_t n = 0; n < 3; ++n) {
            for (int64_t k = 0; k < 9; ++k) {
                StoreReduced(weight, n * 12 + k,
                             static_cast<float>((n + 1) * (k - 4)) * 0.09375F, dtype);
            }
        }

        const auto kernel = PrepareLinearW16Kernel(dtype);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        const Status status = RunLinearEntryWith(*kernel, LinearTestViews{
                                                                  .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
                                                                  .weight_tensor = TensorView{weight, dtype, weight_shape, weight_strides},
                                                                  .output_tensor = MutableTensorView{output.data(), DataType::Float32(), output_shape, output_strides},
                                                          });
        ASSERT_TRUE(status.ok()) << status.ToString();
        for (int64_t row = 0; row < 2; ++row) {
            for (int64_t n = 0; n < 3; ++n) {
                const float expected = FixedFp32Dot(input.data() + row * 13, 1,
                                                    weight + static_cast<size_t>(n * 12) * 2,
                                                    1, 9, dtype);
                EXPECT_EQ(output[static_cast<size_t>(row * 7 + n)], expected);
                const auto [double_expected, absolute_products] =
                        DoubleDotAndAbsoluteProductSum(input.data() + row * 13, 1,
                                                       weight + static_cast<size_t>(n * 12) * 2,
                                                       1, 9, dtype);
                constexpr double unit_roundoff =
                        static_cast<double>(std::numeric_limits<float>::epsilon()) / 2.0;
                const double gamma = (18.0 * unit_roundoff) / (1.0 - 18.0 * unit_roundoff);
                EXPECT_NEAR(output[static_cast<size_t>(row * 7 + n)],
                            static_cast<float>(double_expected),
                            static_cast<float>(gamma * absolute_products + 1.0e-6));
            }
        }
    }
}

TEST(CPUKernelLinearW16, DoesNotRoundFloat32ActivationBeforeMultiplication) {
    constexpr int64_t input_shape[1] = {1};
    constexpr int64_t weight_shape[2] = {1, 1};
    constexpr int64_t output_shape[1] = {1};
    constexpr int64_t strides[2] = {1, 1};
    constexpr int64_t vector_stride[1] = {1};
    const float input = 1.0001F;

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        const uint16_t bits = ReducedBits(1.0F, dtype);
        float output = 0.0F;
        const auto kernel = PrepareLinearW16Kernel(dtype);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        const Status status = RunLinearEntryWith(*kernel, LinearTestViews{
                                                                  .input_tensor = TensorView{&input, DataType::Float32(), input_shape, vector_stride},
                                                                  .weight_tensor = TensorView{&bits, dtype, weight_shape, strides},
                                                                  .output_tensor = MutableTensorView{&output, DataType::Float32(), output_shape, vector_stride},
                                                          });
        ASSERT_TRUE(status.ok()) << status.ToString();
        EXPECT_EQ(output, input);
    }
}

TEST(CPUKernelLinearW16, QuietedReducedNanWeightPropagatesToFloat32Output) {
    constexpr int64_t input_shape[1] = {1};
    constexpr int64_t weight_shape[2] = {1, 1};
    constexpr int64_t vector_stride[1] = {1};
    constexpr int64_t matrix_stride[2] = {1, 1};
    constexpr float input = 1.0F;

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        const uint16_t signaling_nan = dtype == DataType::Float(16) ? 0x7C01U : 0x7F81U;
        float output = 0.0F;
        const auto kernel = PrepareLinearW16Kernel(dtype);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        ASSERT_TRUE(RunLinearEntryWith(*kernel, LinearTestViews{
                                                        .input_tensor = TensorView{&input, DataType::Float32(), input_shape, vector_stride},
                                                        .weight_tensor = TensorView{&signaling_nan, dtype, weight_shape, matrix_stride},
                                                        .output_tensor = MutableTensorView{&output, DataType::Float32(), input_shape, vector_stride},
                                                })
                            .ok());
        EXPECT_TRUE(std::isnan(output));
    }
}

TEST(CPUKernelLinearW16, RejectsOutputAliasAndPreservesZeroKSemantics) {
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    std::array<float, 2> input{};
    std::array<uint16_t, 4> weight{};
    const auto kernel = PrepareLinearW16Kernel(DataType::BFloat(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(RunLinearEntryWith(*kernel, LinearTestViews{
                                                  .input_tensor = TensorView{input.data(), DataType::Float32(), shape, strides},
                                                  .weight_tensor = TensorView{weight.data(), DataType::BFloat(16), shape, strides},
                                                  .output_tensor = MutableTensorView{input.data(), DataType::Float32(), shape, strides},
                                          }),
              StatusCode::kInvalidArgument);

    constexpr int64_t input_zero_shape[2] = {2, 0};
    constexpr int64_t weight_zero_shape[2] = {3, 0};
    constexpr int64_t output_zero_shape[2] = {2, 3};
    constexpr int64_t zero_strides[2] = {1, 1};
    constexpr int64_t output_zero_strides[2] = {3, 1};
    std::array<float, 6> output{};
    output.fill(1.0F);
    const Status zero_status = RunLinearEntryWith(*kernel, LinearTestViews{
                                                                   .input_tensor = TensorView{nullptr, DataType::Float32(), input_zero_shape, zero_strides},
                                                                   .weight_tensor = TensorView{nullptr, DataType::BFloat(16), weight_zero_shape, zero_strides},
                                                                   .output_tensor = MutableTensorView{output.data(), DataType::Float32(), output_zero_shape, output_zero_strides},
                                                           });
    ASSERT_TRUE(zero_status.ok()) << zero_status.ToString();
    for (const float value: output) {
        EXPECT_EQ(value, 0.0F);
    }
}

TEST(CPUKernelLinearW16, RejectsWeightFormatDifferentFromResolvedSelector) {
    constexpr int64_t shape[2] = {1, 1};
    constexpr int64_t strides[2] = {1, 1};
    const float input = 1.0F;
    const uint16_t weight = 0x3f80;
    float output = 42.0F;
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        const auto kernel = PrepareLinearW16Kernel(dtype);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        const DataType wrong = dtype == DataType::Float(16) ? DataType::BFloat(16) : DataType::Float(16);
        const auto status = RunLinearEntryWith(*kernel,
                                               {.input_tensor = TensorView{&input, DataType::Float32(), shape, strides},
                                                .weight_tensor = TensorView{&weight, wrong, shape, strides},
                                                .output_tensor = MutableTensorView{&output, DataType::Float32(), shape, strides}});
        EXPECT_EQ(status, StatusCode::kInvalidArgument);
        EXPECT_EQ(output, 42.0F);
    }
}

TEST(CPUKernelLinearW16, ZeroRowsOrOutputFeaturesAreNoOps) {
    constexpr int64_t strides[2] = {3, 1};
    constexpr int64_t no_rows[2] = {0, 3}, regular_weight[2] = {2, 3}, empty_output[2] = {0, 2};
    constexpr int64_t input_shape[2] = {2, 3}, no_features[2] = {0, 3}, no_feature_output[2] = {2, 0};
    const std::array<uint16_t, 6> weights{};
    const std::array<float, 6> inputs{};
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        const auto kernel = PrepareLinearW16Kernel(dtype);
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        EXPECT_TRUE(RunLinearEntryWith(*kernel,
                                       {.input_tensor = TensorView{nullptr, DataType::Float32(), no_rows, strides},
                                        .weight_tensor = TensorView{weights.data(), dtype, regular_weight, strides},
                                        .output_tensor = MutableTensorView{nullptr, DataType::Float32(), empty_output, strides}})
                            .ok());
        EXPECT_TRUE(RunLinearEntryWith(*kernel,
                                       {.input_tensor = TensorView{inputs.data(), DataType::Float32(), input_shape, strides},
                                        .weight_tensor = TensorView{nullptr, dtype, no_features, strides},
                                        .output_tensor = MutableTensorView{nullptr, DataType::Float32(), no_feature_output, strides}})
                            .ok());
    }
}

TEST(CPUKernelLinear, PreparesPackedIdentityFallback) {
    CpuBackend backend;
    KernelSelector selector = MakeLinearSelector();
    selector.weight_format = WeightFormat::kPacked;
    const auto packed = backend.PrepareKernel(
            OpType::kLinear, selector, OpParams{LinearParams{}});
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    EXPECT_EQ(std::string_view{packed->name},
              "cpu::linear_f32_packed_identity_reference");
    EXPECT_EQ(packed->expected_packing_recipe, CpuIdentityPackingRecipe());
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

TEST(CPUKernelLinear, ReferenceExecutesViewsWithoutSimdAlignmentGuarantee) {
    constexpr int64_t input_shape[2] = {1, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {1, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    alignas(64) std::array<float, 4> input_storage = {0.0F, 1.0F, -2.0F, 0.5F};
    alignas(64) std::array<float, 7> weight_storage = {
            0.0F, 2.0F, 1.0F, -1.0F, -0.5F, 3.0F, 4.0F};
    alignas(64) std::array<float, 3> output_storage = {0.0F, 9.0F, 9.0F};
    const float* const input = input_storage.data() + 1;
    const float* const weight = weight_storage.data() + 1;
    float* const output = output_storage.data() + 1;

    const Status status = RunLinearEntry(LinearTestViews{
            // The views intentionally leave alignment unspecified. The
            // pointers are float-aligned but offset from a 64-byte base, so a
            // later SIMD descriptor must retain an unaligned-compatible path.
            .input_tensor = TensorView{input, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(input, weight, output, 1, 3, 2, 3, 1, 3, 1, 2, 1);
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

TEST(CPUKernelLinear, AllowsDisjointViewsInsideOneAllocation) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    std::array<float, 10> storage{};
    std::array<float, 6> weight{};
    for (int64_t index = 0; index < 6; ++index) {
        storage[static_cast<size_t>(index)] = static_cast<float>(index) * 0.5F - 1.0F;
        weight[static_cast<size_t>(index)] = static_cast<float>(index % 3) * 0.25F + 1.0F;
    }

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{storage.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{storage.data() + 6, DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    ExpectLinearRowsNear(storage.data(), weight.data(), storage.data() + 6, 2, 3, 2, 3, 1, 3, 1, 2, 1);
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

TEST(CPUKernelLinear, ZeroInputFeatureDimensionAllowsOutputAliasingInputs) {
    constexpr int64_t input_shape[2] = {2, 0};
    constexpr int64_t input_strides[2] = {1, 1};
    constexpr int64_t weight_shape[2] = {3, 0};
    constexpr int64_t weight_strides[2] = {1, 1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 1};
    std::array<float, 6> storage{};
    storage.fill(1.0F);

    // in_features == 0 only zero-fills the output and reads neither input nor
    // weight, so sharing one buffer with both is safe here.
    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{storage.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{storage.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{storage.data(), DataType::Float32(), output_shape, output_strides},
    });

    ASSERT_TRUE(status.ok()) << status.ToString();
    for (float value: storage) {
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
    EXPECT_EQ(wrong_dtype, StatusCode::kInvalidArgument) << wrong_dtype.ToString();

    const Status rank_zero = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), {}, {}},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), {}, {}},
    });
    EXPECT_EQ(rank_zero, StatusCode::kInvalidArgument) << rank_zero.ToString();

    constexpr int64_t bad_weight_shape[2] = {2, 3};
    const Status mismatched_shape = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{data, DataType::Float32(), bad_weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(mismatched_shape, StatusCode::kInvalidArgument) << mismatched_shape.ToString();
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
    EXPECT_EQ(invalid_view, StatusCode::kInvalidArgument) << invalid_view.ToString();

    const Status bad_stride = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{data, DataType::Float32(), input_shape, zero_strides},
            .weight_tensor = TensorView{data, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(bad_stride, StatusCode::kInvalidArgument) << bad_stride.ToString();
}

TEST(CPUKernelLinearEntry, RejectsAddressOffsetOverflow) {
    constexpr int64_t weight_shape[2] = {1, 1};
    constexpr int64_t weight_strides[2] = {1, 1};
    float scalar = 1.0F;

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
    EXPECT_EQ(address_overflow, StatusCode::kInvalidArgument) << address_overflow.ToString();
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
    EXPECT_EQ(noncollapsible, StatusCode::kUnimplemented) << noncollapsible.ToString();

    constexpr int64_t input_shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t overlap_shape[2] = {2, 2};
    constexpr int64_t overlap_strides[2] = {1, 1};
    const Status overlap = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{output.data(), DataType::Float32(), overlap_shape, overlap_strides},
    });
    EXPECT_EQ(overlap, StatusCode::kInvalidArgument) << overlap.ToString();
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
    EXPECT_EQ(aliases_input, StatusCode::kInvalidArgument) << aliases_input.ToString();

    const Status aliases_weight = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{weight.data(), DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(aliases_weight, StatusCode::kInvalidArgument) << aliases_weight.ToString();
}

TEST(CPUKernelLinearEntry, RejectsInputOverlapAtDistinctBasePointer) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    std::array<float, 7> storage{};
    std::array<float, 6> weight{};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{storage.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{storage.data() + 1, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelLinearEntry, RejectsWeightOverlapAtDistinctBasePointer) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    std::array<float, 6> input{};
    std::array<float, 7> weight_storage{};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{input.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight_storage.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{weight_storage.data() + 1, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelLinearEntry, ReportsUndecidableColumnStrideOverlapAsUnimplemented) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {7, 2};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {5, 2};
    std::array<float, 13> storage{};
    std::array<float, 6> weight{};

    const Status status = RunLinearEntry(LinearTestViews{
            .input_tensor = TensorView{storage.data(), DataType::Float32(), input_shape, input_strides},
            .weight_tensor = TensorView{weight.data(), DataType::Float32(), weight_shape, weight_strides},
            .output_tensor = MutableTensorView{storage.data() + 1, DataType::Float32(), output_shape, output_strides},
    });

    EXPECT_EQ(status, StatusCode::kUnimplemented) << status.ToString();
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

TEST(CPUKernelLinearW16, PackedIdentityExecutesAndValidatesReducedArtifact) {
    constexpr int64_t input_shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    const float input[6] = {1.0F, -2.0F, 0.5F, 0.25F, 4.0F, -1.5F};
    const float logical[6] = {0.5F, -0.25F, 1.5F, 2.0F, 0.125F, -3.0F};

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        std::array<std::byte, 12> weight_bytes{};
        for (size_t index = 0; index < 6; ++index) {
            StoreReduced(weight_bytes.data(), static_cast<int64_t>(index), logical[index], dtype);
        }
        const TensorView weight_view(weight_bytes.data(), dtype, weight_shape, weight_strides);
        CpuBackend backend;
        KernelSelector selector = MakeLinearSelector();
        selector.weight_dtype = dtype;
        selector.weight_format = WeightFormat::kPacked;
        const auto recipe = backend.GetPackingRecipe(OpType::kLinear, selector);
        ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
        auto packed = backend.PackWeights(
                OpType::kLinear, std::array<TensorView, 1>{weight_view},
                selector, *recipe, test::TestCpuAllocator());
        ASSERT_TRUE(packed.ok()) << packed.status().ToString();
        EXPECT_EQ((*packed)->logical_dtype(), dtype);
        EXPECT_EQ((*packed)->storage().nbytes(), weight_bytes.size());

        const auto kernel = backend.PrepareKernel(
                OpType::kLinear, selector, OpParams{LinearParams{}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        EXPECT_EQ(kernel->name, dtype == DataType::Float(16)
                                        ? "cpu::linear_f32_f16_packed_identity_reference"
                                        : "cpu::linear_f32_bf16_packed_identity_reference");
        const PackedWeightView packed_view{
                .data = (*packed)->storage().data(),
                .nbytes = (*packed)->storage().nbytes(),
                .logical_dtype = (*packed)->logical_dtype(),
                .logical_shape = (*packed)->logical_shape(),
                .recipe_layout = (*packed)->recipe().layout,
                .recipe_alignment = (*packed)->recipe().alignment,
                .alignment = (*packed)->storage().alignment(),
        };

        float output[4]{};
        const TensorView input_view(input, DataType::Float32(), input_shape, input_strides);
        const MutableTensorView output_view(output, DataType::Float32(), output_shape, output_strides);
        PreparedKernelParams prepared;
        ASSERT_TRUE(kernel->params_builder(
                                  KernelParamsBuildContext{
                                          .inputs = std::array<TensorView, 1>{input_view},
                                          .outputs = std::array<MutableTensorView, 1>{output_view},
                                          .attrs = kernel->attrs,
                                          .packed_weight = packed_view,
                                  },
                                  prepared.storage.data())
                            .ok());
        ASSERT_TRUE(RunLinearEntry(*kernel, prepared).ok());
        const auto* packed_bytes = static_cast<const std::byte*>((*packed)->storage().data());
        for (int64_t row = 0; row < 2; ++row) {
            for (int64_t col = 0; col < 2; ++col) {
                EXPECT_EQ(output[row * 2 + col],
                          FixedFp32Dot(input + row * 3, 1,
                                       packed_bytes + static_cast<size_t>(col * 3) * 2,
                                       1, 3, dtype));
            }
        }

        alignas(64) std::array<float, 6> fp32_storage{};
        const PackedWeightView wrong_dtype{
                .data = fp32_storage.data(),
                .nbytes = fp32_storage.size() * sizeof(float),
                .logical_dtype = DataType::Float32(),
                .logical_shape = (*packed)->logical_shape(),
                .recipe_layout = (*packed)->recipe().layout,
                .recipe_alignment = (*packed)->recipe().alignment,
                .alignment = (*packed)->storage().alignment(),
        };
        EXPECT_EQ(kernel->params_builder(
                          KernelParamsBuildContext{
                                  .inputs = std::array<TensorView, 1>{input_view},
                                  .outputs = std::array<MutableTensorView, 1>{output_view},
                                  .attrs = kernel->attrs,
                                  .packed_weight = wrong_dtype,
                          },
                          prepared.storage.data()),
                  StatusCode::kInvalidArgument);
    }
}

} // namespace
