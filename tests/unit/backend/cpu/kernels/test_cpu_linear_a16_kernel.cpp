#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/operators/operator_inference.h"
#include "backend/cpu/kernels/gemm/gemm_a16_w16_internal.h"
#include "backend/cpu/kernels/test_fp_environment_helpers.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

KernelSelector MakeA16Selector(DataType act_dtype, DataType output_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = output_dtype,
            .weight_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareA16Kernel(DataType act_dtype, DataType output_dtype) {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kLinear, MakeA16Selector(act_dtype, output_dtype),
            OpParams{LinearParams{.output_dtype = output_dtype}});
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

uint16_t Bf16Bits(float value) {
    return BFloat16(value).x;
}

float WidenBf16(uint16_t bits) {
    return static_cast<float>(BFloat16(bits, BFloat16::from_bits()));
}

Status RunA16(const ResolvedKernel& kernel,
              DataType act_dtype,
              std::vector<uint16_t>& input,
              std::vector<uint16_t>& weight,
              std::vector<uint16_t>& output16,
              std::vector<float>& output32,
              int64_t rows,
              int64_t in_features,
              int64_t out_features) {
    const std::vector<int64_t> input_shape{rows, in_features};
    const std::vector<int64_t> weight_shape{out_features, in_features};
    const std::vector<int64_t> output_shape{rows, out_features};
    const std::vector<int64_t> input_strides{in_features, 1};
    const std::vector<int64_t> weight_strides{in_features, 1};
    const std::vector<int64_t> output_strides{out_features, 1};
    const bool fp32_output = !output32.empty();
    void* output_data = fp32_output ? static_cast<void*>(output32.data())
                                    : static_cast<void*>(output16.data());
    const DataType output_dtype = fp32_output ? DataType::Float32() : act_dtype;

    const std::array<TensorView, 2> inputs{
            TensorView(input.data(), act_dtype, input_shape, input_strides),
            TensorView(weight.data(), act_dtype, weight_shape, weight_strides)};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output_data, output_dtype, output_shape, output_strides)};
    PreparedParams prepared;
    const Status built = kernel.params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs},
            prepared.storage.data());
    if (!built.ok()) {
        return built;
    }
    return kernel.fn(KernelContext{.kernel_params = prepared.storage.data(), .attrs = kernel.attrs});
}

TEST(LinearA16Kernel, Bf16ProjectionMatchesWidenedReference) {
    const auto kernel = PrepareA16Kernel(DataType::BFloat(16), DataType::BFloat(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    // Exactly representable values keep the expected sums exact.
    std::vector<uint16_t> input_bits{};
    for (const float value: {1.0F, 2.0F, -1.0F, 0.5F, 2.0F, -1.0F, 0.5F, 1.0F}) {
        input_bits.push_back(Bf16Bits(value));
    }
    std::vector<uint16_t> weight_bits{};
    for (const float value: {1.0F, 0.0F, 2.0F, 0.5F,    // row 0
                             -1.0F, 1.0F, 0.0F, 2.0F,   // row 1
                             0.5F, 0.5F, 0.5F, 0.5F}) { // row 2
        weight_bits.push_back(Bf16Bits(value));
    }
    std::vector<uint16_t> output_bits(2 * 3, 0xFFFF);
    std::vector<float> unused;

    const Status status = RunA16(*kernel, DataType::BFloat(16), input_bits, weight_bits,
                                 output_bits, unused, 2, 4, 3);
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Expected values are computed from the widened operands in double.
    const float expected[2][3] = {{-0.75F, 2.0F, 1.25F},
                                  {3.5F, -1.0F, 1.25F}};
    for (int64_t row = 0; row < 2; ++row) {
        for (int64_t col = 0; col < 3; ++col) {
            EXPECT_FLOAT_EQ(WidenBf16(output_bits[row * 3 + col]), expected[row][col])
                    << "row " << row << " col " << col;
        }
    }
}

TEST(LinearA16Kernel, Bf16LmHeadKeepsFloat32Output) {
    const auto kernel = PrepareA16Kernel(DataType::BFloat(16), DataType::Float32());
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    std::vector<uint16_t> input_bits{};
    for (const float value: {1.0F, 2.0F}) {
        input_bits.push_back(Bf16Bits(value));
    }
    std::vector<uint16_t> weight_bits{};
    for (const float value: {3.0F, -1.0F, 0.5F, 0.5F}) {
        weight_bits.push_back(Bf16Bits(value));
    }
    std::vector<uint16_t> unused16;
    std::vector<float> output(2, 0.0F);

    const Status status = RunA16(*kernel, DataType::BFloat(16), input_bits, weight_bits,
                                 unused16, output, 1, 2, 2);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 1.0F);
    EXPECT_FLOAT_EQ(output[1], 1.5F);
}

TEST(LinearA16Kernel, MapsNonUnitFeatureStridesToGemmOperands) {
    const DataType dtype = DataType::BFloat(16);
    const auto kernel = PrepareA16Kernel(dtype, DataType::Float32());
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const std::array<int64_t, 2> input_shape{2, 3};
    const std::array<int64_t, 2> weight_shape{2, 3};
    const std::array<int64_t, 2> output_shape{2, 2};
    const std::array<int64_t, 2> input_strides{8, 2};
    const std::array<int64_t, 2> weight_strides{9, 3};
    const std::array<int64_t, 2> output_strides{7, 2};
    std::array<uint16_t, 13> input;
    std::array<uint16_t, 16> weight;
    input.fill(Bf16Bits(99.0F));
    weight.fill(Bf16Bits(99.0F));
    input[0] = Bf16Bits(1.0F);
    input[2] = Bf16Bits(2.0F);
    input[4] = Bf16Bits(3.0F);
    input[8] = Bf16Bits(4.0F);
    input[10] = Bf16Bits(-1.0F);
    input[12] = Bf16Bits(2.0F);
    weight[0] = Bf16Bits(-1.0F);
    weight[3] = Bf16Bits(0.0F);
    weight[6] = Bf16Bits(2.0F);
    weight[9] = Bf16Bits(0.0F);
    weight[12] = Bf16Bits(1.0F);
    weight[15] = Bf16Bits(2.0F);
    constexpr float guard = -123.0F;
    std::array<float, 10> output;
    output.fill(guard);
    const std::array<TensorView, 2> inputs{
            TensorView(input.data(), dtype, input_shape, input_strides),
            TensorView(weight.data(), dtype, weight_shape, weight_strides)};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::Float32(), output_shape, output_strides)};
    PreparedParams prepared;
    const Status built = kernel->params_builder(
            {.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs}, prepared.storage.data());
    ASSERT_TRUE(built.ok()) << built.ToString();
    ASSERT_TRUE(kernel->fn({.kernel_params = prepared.storage.data(), .attrs = kernel->attrs}).ok());
    const std::array expected{5.0F, guard, 8.0F, guard, guard, guard, guard, 0.0F, guard, 3.0F};
    EXPECT_EQ(output, expected);
}

TEST(LinearA16Kernel, RejectsActivationViewThatDoesNotMatchSelector) {
    const auto kernel = PrepareA16Kernel(DataType::BFloat(16), DataType::BFloat(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    std::vector<uint16_t> input_bits(4, Bf16Bits(1.0F));
    std::vector<uint16_t> weight_bits(4, Bf16Bits(1.0F));
    std::vector<float> output(2, 0.0F);

    const std::vector<int64_t> input_shape{1, 4};
    const std::vector<int64_t> weight_shape{2, 4};
    const std::vector<int64_t> output_shape{1, 2};
    const std::vector<int64_t> strides{4, 1};
    const std::vector<int64_t> output_strides{2, 1};
    const std::array<TensorView, 2> inputs{
            TensorView(input_bits.data(), DataType::Float32(), input_shape, strides),
            TensorView(weight_bits.data(), DataType::BFloat(16), weight_shape, strides)};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::Float32(), output_shape, output_strides)};

    PreparedParams prepared;
    const Status status = kernel->params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs},
            prepared.storage.data());
    EXPECT_EQ(status, StatusCode::kInvalidArgument);
}

TEST(LinearA16Kernel, NativeBf16SafetyDomainRejectsUnderflowAndCancellation) {
    // The production guard is baseline code, so its underflow domain is tested
    // even when this host cannot execute the optional BF16 instruction.
    const auto safe = [](uint16_t input, uint16_t weight) {
        return cpu::detail::IsBf16NativeDotProductSafe(
                reinterpret_cast<const std::byte*>(&input),
                reinterpret_cast<const std::byte*>(&weight), 1);
    };
    EXPECT_TRUE(safe(0x3F80, 0x3F00));  // 1 * 0.5
    EXPECT_TRUE(safe(0x8000, 0x0001));  // signed zero * finite subnormal
    EXPECT_TRUE(safe(0x2380, 0x2380));  // lower safe exponent-sum boundary
    EXPECT_FALSE(safe(0x0001, 0x7F00)); // subnormal * large normal
    EXPECT_FALSE(safe(0x1F00, 0x1F00)); // normal * normal -> subnormal
    EXPECT_FALSE(safe(0x2301, 0x2381)); // normal products may cancel below FLT_MIN
    EXPECT_FALSE(safe(0x0000, 0x7F80)); // 0 * infinity must retain NaN behavior
    EXPECT_FALSE(safe(0x7FC1, 0x3F80)); // NaN stays on the reference path
    EXPECT_FALSE(safe(0x7F00, 0x7F00)); // exceptional product magnitude

    const std::array<uint16_t, 2> input{0x3F80, 0x0001};
    const std::array<uint16_t, 2> weight{0x3F80, 0x7F00};
    EXPECT_FALSE(cpu::detail::IsBf16NativeDotProductSafe(
            reinterpret_cast<const std::byte*>(input.data()),
            reinterpret_cast<const std::byte*>(weight.data()), 2));
}

TEST(LinearA16Kernel, Bf16UnderflowAgreesAcrossVectorBodyAndTail) {
    test::ScopedNearestGradualUnderflow environment;
    ASSERT_TRUE(environment.ok());
    struct DotCase {
        std::array<uint16_t, 2> input;
        std::array<uint16_t, 2> weight;
        uint32_t expected_f32;
        uint16_t expected_bf16;
    };
    // Literal expectations are independent of the production conversion helper.
    const std::array cases{
            DotCase{{0x0001, 0}, {0x7F00, 0}, 0x3C800000U, 0x3C80}, // 2^-133 * 2^127
            DotCase{{0x1F00, 0}, {0x1F00, 0}, 0x00080000U, 0x0008}, // 2^-65 * 2^-65
            // Both products are normal; their exact difference is 2^-127.
            DotCase{{0x2301, 0xA300}, {0x2381, 0x2382}, 0x00400000U, 0x0040},
    };
    struct Placement {
        int64_t in_features;
        size_t first;
    };
    constexpr std::array placements{
            Placement{32, 0}, Placement{34, 32}, Placement{64, 32}, Placement{66, 64}};
    CpuBackend reference_backend(CpuCapabilities{});
    CpuBackend host_backend;
    for (auto* backend: {&reference_backend, &host_backend}) {
        for (const auto output_dtype: {DataType::BFloat(16), DataType::Float32()}) {
            auto kernel = backend->PrepareKernel(
                    OpType::kLinear, MakeA16Selector(DataType::BFloat(16), output_dtype),
                    OpParams{LinearParams{.output_dtype = output_dtype}});
            ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
            SCOPED_TRACE(std::string(kernel->name));
            for (const auto& example: cases) {
                for (const auto placement: placements) {
                    SCOPED_TRACE(placement.in_features);
                    SCOPED_TRACE(placement.first);
                    std::vector<uint16_t> input(placement.in_features, 0);
                    std::vector<uint16_t> weight(placement.in_features, 0);
                    for (size_t index = 0; index < 2; ++index) {
                        input[placement.first + index] = example.input[index];
                        weight[placement.first + index] = example.weight[index];
                    }
                    std::vector<uint16_t> output16(output_dtype == DataType::BFloat(16) ? 1 : 0);
                    std::vector<float> output32(output_dtype == DataType::Float32() ? 1 : 0);
                    ASSERT_TRUE(RunA16(*kernel, DataType::BFloat(16), input, weight,
                                       output16, output32, 1, placement.in_features, 1)
                                        .ok());
                    if (output_dtype == DataType::Float32()) {
                        EXPECT_EQ(std::bit_cast<uint32_t>(output32[0]), example.expected_f32);
                    } else {
                        EXPECT_EQ(output16[0], example.expected_bf16);
                    }
                }
            }
        }
    }
}

TEST(LinearA16Kernel, Bf16UnsafeLaterRowPreservesEarlierOutput) {
    test::ScopedNearestGradualUnderflow environment;
    ASSERT_TRUE(environment.ok());
    auto kernel = PrepareA16Kernel(DataType::BFloat(16), DataType::Float32());
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    std::vector<uint16_t> input(64, 0);
    std::vector<uint16_t> weight(32, 0);
    input[0] = 0x6000;  // 2^65
    input[32] = 0x1F00; // 2^-65
    weight[0] = 0x1F00;
    std::vector<uint16_t> unused;
    std::vector<float> output(2, -1.0F);
    ASSERT_TRUE(RunA16(*kernel, DataType::BFloat(16), input, weight,
                       unused, output, 2, 32, 1)
                        .ok());
    EXPECT_EQ(std::bit_cast<uint32_t>(output[0]), 0x3F800000U);
    EXPECT_EQ(std::bit_cast<uint32_t>(output[1]), 0x00080000U);
}

TEST(LinearA16Kernel, PackedIdentityZeroInputFeatureDimensionWritesZero) {
    for (const DataType act_dtype: {DataType::BFloat(16), DataType::Float(16)}) {
        CpuBackend backend;
        KernelSelector selector = MakeA16Selector(act_dtype, act_dtype);
        selector.weight_format = WeightFormat::kPacked;
        const auto kernel = backend.PrepareKernel(
                OpType::kLinear, selector, OpParams{LinearParams{}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

        const std::vector<int64_t> input_shape{1, 0};
        const std::vector<int64_t> input_strides{0, 1};
        const std::vector<int64_t> packed_shape{3, 0};
        const std::vector<int64_t> output_shape{1, 3};
        const std::vector<int64_t> output_strides{3, 1};
        std::array<uint16_t, 3> output{0xFFFFU, 0xFFFFU, 0xFFFFU};

        // K == 0 reads neither input nor weight, so both empty buffers may
        // legally carry null data pointers; only the output must be present.
        const TensorView input{nullptr, act_dtype, input_shape, input_strides};
        ASSERT_TRUE(input.is_valid());
        const PackedWeightView packed{
                .data = nullptr,
                .nbytes = 0,
                .logical_dtype = act_dtype,
                .logical_shape = packed_shape,
                .recipe_layout = cpu::kCpuIdentityPackingLayout,
                .recipe_alignment = cpu::kCpuIdentityPackingAlignment,
                .alignment = cpu::kCpuIdentityPackingAlignment,
        };
        PreparedParams prepared;
        const Status built = kernel->params_builder(
                KernelParamsBuildContext{
                        .inputs = std::array<TensorView, 1>{input},
                        .outputs = std::array<MutableTensorView, 1>{
                                MutableTensorView{output.data(), act_dtype, output_shape, output_strides}},
                        .attrs = kernel->attrs,
                        .packed_weight = packed,
                },
                prepared.storage.data());
        ASSERT_TRUE(built.ok()) << built.ToString();
        const Status ran = kernel->fn(KernelContext{
                .kernel_params = prepared.storage.data(),
                .attrs = kernel->attrs});
        ASSERT_TRUE(ran.ok()) << ran.ToString();
        for (uint16_t bits: output) {
            EXPECT_EQ(bits, 0U);
        }
    }
}

struct A16BlockedShape {
    int64_t rows;
    int64_t in_features;
    int64_t out_features;
    bool fp32_output;
};

std::string A16BlockedShapeName(const testing::TestParamInfo<A16BlockedShape>& info) {
    return "M" + std::to_string(info.param.rows) + "K" +
           std::to_string(info.param.in_features) + "N" +
           std::to_string(info.param.out_features) +
           (info.param.fp32_output ? "F32Out" : "Bf16Out");
}

class LinearA16KernelBlockedTest : public testing::TestWithParam<A16BlockedShape> {};

TEST_P(LinearA16KernelBlockedTest, BlockedPathMatchesWidenedReference) {
    const A16BlockedShape test_case = GetParam();
    const DataType output_dtype = test_case.fp32_output ? DataType::Float32() : DataType::BFloat(16);
    const auto kernel = PrepareA16Kernel(DataType::BFloat(16), output_dtype);
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const auto value = [](size_t index) {
        return static_cast<float>(static_cast<int64_t>((index * 17U + 3U) % 59U) - 29) * 0.03125F;
    };
    std::vector<uint16_t> input_bits(static_cast<size_t>(test_case.rows * test_case.in_features));
    std::vector<uint16_t> weight_bits(static_cast<size_t>(test_case.out_features * test_case.in_features));
    for (size_t i = 0; i < input_bits.size(); ++i) {
        input_bits[i] = Bf16Bits(value(i + 5));
    }
    for (size_t i = 0; i < weight_bits.size(); ++i) {
        weight_bits[i] = Bf16Bits(value(i));
    }
    std::vector<uint16_t> output16;
    std::vector<float> output32;
    if (test_case.fp32_output) {
        output32.resize(static_cast<size_t>(test_case.rows * test_case.out_features));
    } else {
        output16.resize(static_cast<size_t>(test_case.rows * test_case.out_features), 0xFFFFU);
    }

    const Status status = RunA16(*kernel, DataType::BFloat(16), input_bits, weight_bits,
                                 output16, output32, test_case.rows,
                                 test_case.in_features, test_case.out_features);
    ASSERT_TRUE(status.ok()) << status.ToString();

    // Oracle accumulates the widened operands in double; the BF16 output
    // contract additionally rounds once at the end.
    for (int64_t row = 0; row < test_case.rows; ++row) {
        for (int64_t col = 0; col < test_case.out_features; ++col) {
            double expected = 0.0;
            double absolute_products = 0.0;
            for (int64_t k = 0; k < test_case.in_features; ++k) {
                const double product = static_cast<double>(WidenBf16(input_bits[static_cast<size_t>(row * test_case.in_features + k)])) *
                                       static_cast<double>(WidenBf16(weight_bits[static_cast<size_t>(col * test_case.in_features + k)]));
                expected += product;
                absolute_products += std::fabs(product);
            }
            const float rounded_expected = test_case.fp32_output
                                                   ? static_cast<float>(expected)
                                                   : WidenBf16(Bf16Bits(static_cast<float>(expected)));
            const float actual = test_case.fp32_output
                                         ? output32[static_cast<size_t>(row * test_case.out_features + col)]
                                         : WidenBf16(output16[static_cast<size_t>(row * test_case.out_features + col)]);
            const double ku = 2.0 * static_cast<double>(test_case.in_features) *
                              static_cast<double>(std::numeric_limits<float>::epsilon()) * 0.5;
            EXPECT_NEAR(actual, rounded_expected,
                        ku / (1.0 - ku) * absolute_products + 1.0e-6)
                    << "row=" << row << " col=" << col;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        BlockedBoundaries,
        LinearA16KernelBlockedTest,
        testing::Values(
                // 6/4/2-row tile edges inside one 48-row scratch block.
                A16BlockedShape{9, 8, 16, false},
                A16BlockedShape{15, 512, 33, false},
                A16BlockedShape{47, 512, 33, true},
                A16BlockedShape{48, 512, 33, true},
                A16BlockedShape{49, 512, 33, false},
                A16BlockedShape{63, 512, 129, true},
                A16BlockedShape{65, 512, 129, true},
                A16BlockedShape{17, 513, 17, false}),
        A16BlockedShapeName);

} // namespace
