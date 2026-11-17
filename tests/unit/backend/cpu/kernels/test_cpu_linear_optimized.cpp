#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "backend/cpu/kernels/linear/linear_internal.h"
#include "test_utils/test_allocator.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <tuple>
#include <vector>

namespace {
using namespace aethermind;

struct Shape {
    int64_t m;
    int64_t k;
    int64_t n;
};

float Value(size_t index) {
    return static_cast<float>(static_cast<int>((index * 17U + 3U) % 59U) - 29) * 0.03125F;
}

float DecodeWeight(const std::byte* data, DataType dtype) {
    if (dtype == DataType::Float32()) {
        float value;
        std::memcpy(&value, data, sizeof(value));
        return value;
    }
    uint16_t bits;
    std::memcpy(&bits, data, sizeof(bits));
    const int fraction_bits = dtype == DataType::Float(16) ? 10 : 7;
    const int bias = fraction_bits == 10 ? 15 : 127;
    const unsigned exponent = (bits >> fraction_bits) & (fraction_bits == 10 ? 31U : 255U);
    const unsigned fraction = bits & ((1U << fraction_bits) - 1U);
    const double significand = exponent == 0 ? fraction : (1U << fraction_bits) + fraction;
    const float magnitude = static_cast<float>(std::ldexp(significand, static_cast<int>(exponent == 0 ? 1 : exponent) - bias - fraction_bits));
    return (bits & 0x8000U) != 0 ? -magnitude : magnitude;
}

void WriteWeight(std::byte* destination, DataType dtype, float value) {
    if (dtype == DataType::Float32()) {
        std::memcpy(destination, &value, sizeof(value));
    } else {
        const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
        std::memcpy(destination, &bits, sizeof(bits));
    }
}

using Case = std::tuple<int, bool, bool, Shape>;
class CpuLinearOptimized : public testing::TestWithParam<Case> {};

TEST_P(CpuLinearOptimized, PreparedPathMatchesIndependentOracleAndPreservesGuards) {
    const auto [precision, packed, strided, shape] = GetParam();
    const DataType dtype = precision == 0 ? DataType::Float32() : precision == 1 ? DataType::Float(16)
                                                                                 : DataType::BFloat(16);
    const int64_t input_col = strided ? 2 : 1;
    const int64_t output_col = strided ? 3 : 1;
    const int64_t weight_col = strided && !packed ? 2 : 1;
    const int64_t input_row = shape.k * input_col + 3;
    const int64_t output_row = shape.n * output_col + 5;
    const int64_t weight_row = shape.k * weight_col + (packed ? 0 : 3);
    const size_t element_bytes = static_cast<size_t>(dtype.nbytes());
    const size_t prefix = dtype == DataType::Float32() ? sizeof(float) : 1;
    std::vector<float> input(static_cast<size_t>(shape.m * input_row) + 2, 0.0F);
    std::vector<float> output(static_cast<size_t>(shape.m * output_row) + 2, -123.0F);
    std::vector<std::byte> storage(prefix + static_cast<size_t>(shape.n * weight_row) * element_bytes + 8);
    std::byte* weight = storage.data() + prefix;
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = Value(i + 5);
    }
    for (int64_t n = 0; n < shape.n; ++n) {
        for (int64_t k = 0; k < shape.k; ++k) {
            WriteWeight(weight + static_cast<size_t>(n * weight_row + k * weight_col) * element_bytes, dtype, Value(static_cast<size_t>(n * shape.k + k)));
        }
    }
    const std::array<int64_t, 2> input_shape{shape.m, shape.k};
    const std::array<int64_t, 2> weight_shape{shape.n, shape.k};
    const std::array<int64_t, 2> output_shape{shape.m, shape.n};
    const std::array<int64_t, 2> input_strides{input_row, input_col};
    const std::array<int64_t, 2> weight_strides{weight_row, weight_col};
    const std::array<int64_t, 2> output_strides{output_row, output_col};
    const TensorView weights(weight, dtype, weight_shape, weight_strides);
    const std::array<TensorView, 2> inputs{
            TensorView(input.data(), DataType::Float32(), input_shape, input_strides), weights};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::Float32(), output_shape, output_strides)};
    CpuBackend backend;
    const KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .output_dtype = DataType::Float32(),
            .weight_dtype = dtype,
            .weight_format = packed ? WeightFormat::kPacked : WeightFormat::kPlain,
            .phase = shape.m == 1 ? ExecPhase::kDecode : ExecPhase::kPrefill,
    };
    const auto resolved = backend.PrepareKernel(OpType::kLinear, selector, OpParams{LinearParams{}});
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    std::unique_ptr<PackedWeight> artifact;
    std::optional<PackedWeightView> packed_view;
    if (packed) {
        const std::array<TensorView, 1> components{weights};
        auto result = backend.PackWeights(OpType::kLinear, components, selector,
                                          resolved->expected_packing_recipe, test::TestCpuAllocator());
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        artifact = std::move(*result);
        ASSERT_EQ(artifact->recipe(), CpuIdentityPackingRecipe());
        packed_view = PackedWeightView{
                .data = artifact->storage().data(),
                .nbytes = artifact->storage().nbytes(),
                .logical_dtype = dtype,
                .logical_shape = artifact->logical_shape(),
                .recipe_layout = artifact->recipe().layout,
                .recipe_alignment = artifact->recipe().alignment,
                .alignment = artifact->storage().alignment(),
        };
    }
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> params{};
    const auto kernel_inputs = packed ? std::span<const TensorView>(inputs.data(), 1) : std::span<const TensorView>(inputs);
    const Status built = resolved->params_builder({.inputs = kernel_inputs, .outputs = outputs, .packed_weight = packed_view}, params.data());
    ASSERT_TRUE(built.ok()) << built.ToString();
    ASSERT_TRUE(resolved->fn({.kernel_params = params.data()}).ok());
    std::vector<bool> written(output.size(), false);
    for (int64_t m = 0; m < shape.m; ++m) {
        for (int64_t n = 0; n < shape.n; ++n) {
            double expected = 0.0;
            double absolute_products = 0.0;
            for (int64_t k = 0; k < shape.k; ++k) {
                const double product = static_cast<double>(input[static_cast<size_t>(m * input_row + k * input_col)]) *
                                       DecodeWeight(weight + static_cast<size_t>(n * weight_row + k * weight_col) * element_bytes, dtype);
                expected += product;
                absolute_products += std::fabs(product);
            }
            const size_t index = static_cast<size_t>(m * output_row + n * output_col);
            const double ku = 2.0 * static_cast<double>(shape.k) * std::numeric_limits<float>::epsilon() * 0.5;
            EXPECT_NEAR(output[index], expected, ku / (1.0 - ku) * absolute_products + 1.0e-6) << resolved->name;
            if (shape.k == 0) {
                EXPECT_FALSE(std::signbit(output[index]));
            }
            written[index] = true;
        }
    }
    for (size_t i = 0; i < output.size(); ++i) {
        if (!written[i]) {
            EXPECT_EQ(output[i], -123.0F) << "guard index=" << i;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        PrecisionLayoutAndTails, CpuLinearOptimized,
        testing::Combine(testing::Values(0, 1, 2), testing::Bool(), testing::Bool(),
                         testing::Values(Shape{1, 1, 3}, Shape{1, 7, 5}, Shape{1, 9, 19}, Shape{2, 33, 19},
                                         Shape{4, 513, 35}, Shape{8, 32, 16}, Shape{9, 33, 19}, Shape{11, 513, 35},
                                         Shape{49, 17, 19}, Shape{32, 32, 33}, Shape{2, 0, 7}, Shape{0, 7, 5}, Shape{2, 7, 0})));

class CpuLinearOptimizedSafety : public testing::TestWithParam<bool> {};

TEST_P(CpuLinearOptimizedSafety, ExtremeMagnitudeAndNonFiniteValuesUseFullReferenceFallback) {
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    const bool f16 = GetParam();
    const DataType dtype = f16 ? DataType::Float(16) : DataType::BFloat(16);
    auto runner = &cpu::detail::RunLinearF32Bf16Avx2Fma;
    CpuBackend backend;
    if (f16) {
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
        if (!backend.cpu_capabilities().effective_features.Contains(CpuFeature::kF16c)) {
            GTEST_SKIP() << "F16C is required for FP16 SIMD execution";
        }
        runner = &cpu::detail::RunLinearF32F16Avx2Fma;
#else
        GTEST_SKIP() << "FP16 SIMD was not compiled";
#endif
    }
    if (!backend.cpu_capabilities().effective_features.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}))) {
        GTEST_SKIP() << "AVX2/FMA execution requires actual hardware";
    }
    for (const int64_t m: {1, 11}) {
        constexpr int64_t k = 33;
        constexpr int64_t n = 19;
        for (const float special: {std::numeric_limits<float>::max(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) {
            std::vector<float> input(static_cast<size_t>(m * k), special);
            std::vector<std::byte> weights(static_cast<size_t>(n * k) * 2);
            std::vector<float> output(static_cast<size_t>(m * n));
            std::vector<float> expected(output.size());
            for (int64_t i = 0; i < n * k; ++i) {
                WriteWeight(weights.data() + i * 2, dtype, i % 2 == 0 ? 2.0F : -2.0F);
            }
            cpu::detail::LinearA32W16KernelArgs args{
                    .input = input.data(),
                    .weight = weights.data(),
                    .output = output.data(),
                    .weight_dtype = dtype,
                    .row_count = m,
                    .in_features = k,
                    .out_features = n,
                    .input_row_stride = k,
                    .weight_row_stride = k,
                    .output_row_stride = n};
            ASSERT_TRUE(runner(args).ok());
            args.output = expected.data();
            ASSERT_TRUE(cpu::detail::RunLinearF32W16Reference(args).ok());
            for (size_t i = 0; i < output.size(); ++i) {
                if (std::isnan(expected[i])) {
                    EXPECT_TRUE(std::isnan(output[i]));
                } else {
                    EXPECT_EQ(output[i], expected[i]);
                }
            }
        }
    }
#else
    GTEST_SKIP() << "W16 SIMD was not compiled";
#endif
}

TEST_P(CpuLinearOptimizedSafety, ProductOverflowCannotBeHiddenByFusedCancellation) {
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
    const bool f16 = GetParam();
    const DataType dtype = f16 ? DataType::Float(16) : DataType::BFloat(16);
    auto runner = &cpu::detail::RunLinearF32Bf16Avx2Fma;
    CpuBackend backend;
    if (f16) {
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
        if (!backend.cpu_capabilities().effective_features.Contains(CpuFeature::kF16c)) {
            GTEST_SKIP() << "F16C is required for FP16 SIMD execution";
        }
        runner = &cpu::detail::RunLinearF32F16Avx2Fma;
#else
        GTEST_SKIP() << "FP16 SIMD was not compiled";
#endif
    }
    if (!backend.cpu_capabilities().effective_features.ContainsAll(CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}))) {
        GTEST_SKIP() << "AVX2/FMA execution requires actual hardware";
    }
    for (const int64_t m: {1, 11}) {
        constexpr int64_t k = 33, n = 19;
        std::vector<float> input(static_cast<size_t>(m * k), 0.0F);
        std::vector<std::byte> weights(static_cast<size_t>(n * k) * 2);
        std::vector<float> output(static_cast<size_t>(m * n));
        for (int64_t row = 0; row < m; ++row) {
            input[static_cast<size_t>(row * k)] = -std::numeric_limits<float>::max();
            input[static_cast<size_t>(row * k + 8)] = std::numeric_limits<float>::max();
        }
        for (int64_t col = 0; col < n; ++col) {
            WriteWeight(weights.data() + col * k * 2, dtype, 1.0F);
            WriteWeight(weights.data() + (col * k + 8) * 2, dtype, 2.0F);
        }
        // Scalar product rounding overflows before the addition, while an FMA
        // could cancel its unrounded product back to a finite value.
        cpu::detail::LinearA32W16KernelArgs args{
                .input = input.data(), .weight = weights.data(), .output = output.data(), .weight_dtype = dtype,
                .row_count = m, .in_features = k, .out_features = n,
                .input_row_stride = k, .weight_row_stride = k, .output_row_stride = n};
        ASSERT_TRUE(runner(args).ok());
        for (const float value: output) {
            EXPECT_EQ(value, std::numeric_limits<float>::infinity());
        }
    }
#else
    GTEST_SKIP() << "W16 SIMD was not compiled";
#endif
}

INSTANTIATE_TEST_SUITE_P(WeightPrecision, CpuLinearOptimizedSafety, testing::Bool());

TEST(CpuLinearOptimizedDispatch, F16cIsRequiredOnlyForFp16WeightsAndRecipesRemainIdentity) {
    const auto capabilities = [](bool f16c) {
        auto features = CpuFeatureSet::From({CpuFeature::kAvx, CpuFeature::kAvx2, CpuFeature::kFma});
        if (f16c) {
            features.Enable(CpuFeature::kF16c);
        }
        return CpuCapabilities{.architecture = CpuArchitecture::kX86_64,
                               .usable_features = features,
                               .effective_features = features};
    };
    for (const bool f16c: {false, true}) {
        CpuBackend backend(capabilities(f16c));
        for (const bool packed: {false, true}) {
            for (const bool bf16: {false, true}) {
                const KernelSelector selector{
                        .device_type = DeviceType::kCPU,
                        .act_dtype = DataType::Float32(),
                        .output_dtype = DataType::Float32(),
                        .weight_dtype = bf16 ? DataType::BFloat(16) : DataType::Float(16),
                        .weight_format = packed ? WeightFormat::kPacked : WeightFormat::kPlain,
                        .phase = ExecPhase::kDecode};
                const auto kernel = backend.PrepareKernel(OpType::kLinear, selector, OpParams{LinearParams{}});
                ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
                bool optimized = false;
#if defined(GEMM_HAS_AVX2_FMA_KERNEL)
                optimized = bf16;
#endif
#if defined(GEMM_HAS_A32_F16_AVX2_FMA_KERNEL)
                optimized = optimized || f16c;
#endif
                const std::string name = std::string("cpu::linear_f32_") + (bf16 ? "wbf16" : "wf16") +
                                         (packed ? "_packed_identity" : "") + (optimized ? "_avx2_fma" : "_reference");
                EXPECT_EQ(kernel->name, name);
                if (packed) {
                    EXPECT_EQ(kernel->expected_packing_recipe, CpuIdentityPackingRecipe());
                }
            }
        }
    }
}

} // namespace
