#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/gemm/gemm_a16_w16_internal.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using aethermind::cpu::DetectCpuCapabilities;

CpuCapabilities Synthetic(CpuFeatureSet effective_features) {
    return CpuCapabilities{
            .architecture = CpuArchitecture::kX86_64,
            .usable_features = effective_features,
            .effective_features = effective_features,
    };
}

KernelSelector MakeSelector(DataType act_dtype, DataType output_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = output_dtype,
            .weight_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

OpParams MakeParams(DataType act_dtype, DataType output_dtype) {
    if (output_dtype == act_dtype) {
        return OpParams{LinearParams{}};
    }
    return OpParams{LinearParams{.output_dtype = output_dtype}};
}

std::string ResolvedName(const CpuCapabilities& capabilities,
                         DataType act_dtype,
                         DataType output_dtype) {
    CpuBackend backend(capabilities);
    const auto kernel = backend.PrepareKernel(OpType::kLinear,
                                              MakeSelector(act_dtype, output_dtype),
                                              MakeParams(act_dtype, output_dtype));
    if (!kernel.ok()) {
        return kernel.status().ToString();
    }
    return std::string{kernel->name};
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

void WriteA16(std::byte* destination, float value, DataType dtype, size_t index) {
    const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
    std::memcpy(destination + index * 2, &bits, sizeof(bits));
}

float ReadA16(const std::byte* source, DataType dtype, size_t index) {
    uint16_t bits = 0;
    std::memcpy(&bits, source + index * 2, sizeof(bits));
    return dtype == DataType::Float(16) ? static_cast<float>(Half(bits, Half::from_bits()))
                                        : static_cast<float>(BFloat16(bits, BFloat16::from_bits()));
}

TEST(LinearA16Dispatch, ReferenceSelectedWithoutVectorFeatures) {
    const auto capabilities = Synthetic(CpuFeatureSet{});
    EXPECT_EQ(ResolvedName(capabilities, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_reference");
    EXPECT_EQ(ResolvedName(capabilities, DataType::Float(16), DataType::Float32()),
              "cpu::linear_f16_wf16_out32_reference");
}

TEST(LinearA16Dispatch, Avx2FmaSelectedWhenWideningFeaturesArePresent) {
    const auto capabilities = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx, CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}));
    EXPECT_EQ(ResolvedName(capabilities, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_avx2_fma");
    EXPECT_EQ(ResolvedName(capabilities, DataType::Float(16), DataType::Float(16)),
              "cpu::linear_f16_wf16_avx2_fma");
}

TEST(LinearA16Dispatch, F16Avx2RequiresF16cAndFallsBackWithoutIt) {
    // The F16 widening fallback uses F16C (_mm256_cvtph_ps); without it the F16
    // AVX2 kernel is ineligible and resolution stays on the scalar reference.
    // BF16 widens with an exact integer shift, so it remains eligible.
    const auto no_f16c = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx, CpuFeature::kAvx2, CpuFeature::kFma}));
    EXPECT_EQ(ResolvedName(no_f16c, DataType::Float(16), DataType::Float(16)),
              "cpu::linear_f16_wf16_reference");
    EXPECT_EQ(ResolvedName(no_f16c, DataType::Float(16), DataType::Float32()),
              "cpu::linear_f16_wf16_out32_reference");
    EXPECT_EQ(ResolvedName(no_f16c, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_avx2_fma");
}

TEST(LinearA16Dispatch, Avx512Bf16NativeNeedsBothAvx512Features) {
    // AVX-512F alone cannot use VDPBF16PS, so the widening fallback stays in
    // place; adding AVX512-BF16 promotes the native kernel.
    const auto partial = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c,
             CpuFeature::kAvx512F}));
    EXPECT_EQ(ResolvedName(partial, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_avx2_fma");

    const auto native = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c,
             CpuFeature::kAvx512F, CpuFeature::kAvx512Bf16}));
    EXPECT_EQ(ResolvedName(native, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_avx512_bf16");
    EXPECT_EQ(ResolvedName(native, DataType::BFloat(16), DataType::Float32()),
              "cpu::linear_bf16_wbf16_out32_avx512_bf16");
    // The FP16 native kernel is not eligible from BF16-only AVX-512 support.
    EXPECT_EQ(ResolvedName(native, DataType::Float(16), DataType::Float(16)),
              "cpu::linear_f16_wf16_avx2_fma");
}

TEST(LinearA16Dispatch, Avx512Fp16NativeNeedsFp16Feature) {
    const auto native = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c,
             CpuFeature::kAvx512F, CpuFeature::kAvx512Fp16}));
    EXPECT_EQ(ResolvedName(native, DataType::Float(16), DataType::Float(16)),
              "cpu::linear_f16_wf16_avx512_fp16");
    EXPECT_EQ(ResolvedName(native, DataType::Float(16), DataType::Float32()),
              "cpu::linear_f16_wf16_out32_avx512_fp16");
    EXPECT_EQ(ResolvedName(native, DataType::BFloat(16), DataType::BFloat(16)),
              "cpu::linear_bf16_wbf16_avx2_fma");
}

TEST(LinearA16Dispatch, PackedIdentitySelectsTheVectorKernel) {
    const auto capabilities = Synthetic(CpuFeatureSet::From(
            {CpuFeature::kAvx, CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}));
    CpuBackend backend(capabilities);
    KernelSelector selector = MakeSelector(DataType::BFloat(16), DataType::BFloat(16));
    selector.weight_format = WeightFormat::kPacked;
    const auto kernel = backend.PrepareKernel(OpType::kLinear, selector,
                                              OpParams{LinearParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string{kernel->name}, "cpu::linear_bf16_wbf16_packed_identity_avx2_fma");
}

// Numeric evidence for the AVX2/FMA widening fallback. The native AVX-512
// kernels use the same lane-reduction contract but require AVX-512 hardware to
// execute; their numeric evidence is collected on the first hardware run.
TEST(LinearA16Avx2Fma, MatchesDoublePrecisionOracleWithTailElements) {
    const auto detected = DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();
    const auto required = CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c});
    if (!detected->effective_features.ContainsAll(required)) {
        GTEST_SKIP() << "AVX2/FMA/F16C are required to execute the fallback";
    }

    constexpr int64_t kRows = 2;
    constexpr int64_t kIn = 10; // 8-wide body plus a two-element tail
    constexpr int64_t kOut = 3;
    constexpr std::array<float, kIn> input_values{
            0.5F, -1.25F, 2.0F, 0.75F, -0.5F, 1.5F, -2.0F, 0.25F, 1.0F, -0.75F};
    constexpr std::array<float, kOut * kIn> weight_values{
            1.0F, 0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F, -1.5F, 0.75F, 1.0F,
            -1.0F, 2.0F, 1.25F, -0.25F, 0.5F, 1.0F, -2.0F, 0.5F, -0.5F, 1.5F,
            0.25F, -1.5F, 1.0F, 0.5F, 2.0F, -1.0F, 0.75F, 1.25F, -0.25F, 0.5F};

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        CpuBackend backend(Synthetic(required));
        const auto kernel = backend.PrepareKernel(OpType::kLinear, MakeSelector(dtype, dtype),
                                                  OpParams{LinearParams{}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        ASSERT_EQ(std::string{kernel->name},
                  dtype == DataType::Float(16) ? "cpu::linear_f16_wf16_avx2_fma"
                                               : "cpu::linear_bf16_wbf16_avx2_fma");

        std::vector<std::byte> input(static_cast<size_t>(kRows * kIn) * 2);
        std::vector<std::byte> weight(static_cast<size_t>(kOut * kIn) * 2);
        std::vector<std::byte> output(static_cast<size_t>(kRows * kOut) * 2);
        for (size_t index = 0; index < input.size() / 2; ++index) {
            WriteA16(input.data(), input_values[index % input_values.size()], dtype, index);
        }
        for (size_t index = 0; index < weight.size() / 2; ++index) {
            WriteA16(weight.data(), weight_values[index], dtype, index);
        }

        const std::array<int64_t, 2> input_shape{kRows, kIn};
        const std::array<int64_t, 2> weight_shape{kOut, kIn};
        const std::array<int64_t, 2> output_shape{kRows, kOut};
        const std::array<int64_t, 2> input_strides{kIn, 1};
        const std::array<int64_t, 2> weight_strides{kIn, 1};
        const std::array<int64_t, 2> output_strides{kOut, 1};
        const std::array<TensorView, 2> inputs{
                TensorView(input.data(), dtype, input_shape, input_strides),
                TensorView(weight.data(), dtype, weight_shape, weight_strides)};
        const std::array<MutableTensorView, 1> outputs{
                MutableTensorView(output.data(), dtype, output_shape, output_strides)};

        PreparedParams prepared;
        const Status built = kernel->params_builder(
                KernelParamsBuildContext{.inputs = inputs,
                                         .outputs = outputs,
                                         .attrs = kernel->attrs},
                prepared.storage.data());
        ASSERT_TRUE(built.ok()) << built.ToString();
        const Status ran = kernel->fn({.kernel_params = prepared.storage.data(),
                                       .attrs = kernel->attrs});
        ASSERT_TRUE(ran.ok()) << ran.ToString();

        for (int64_t row = 0; row < kRows; ++row) {
            for (int64_t col = 0; col < kOut; ++col) {
                double expected = 0.0;
                for (int64_t k = 0; k < kIn; ++k) {
                    const double input_value = ReadA16(
                            input.data(), dtype,
                            static_cast<size_t>(row * kIn + k));
                    const double weight_value = ReadA16(
                            weight.data(), dtype,
                            static_cast<size_t>(col * kIn + k));
                    expected += input_value * weight_value;
                }
                const double actual = ReadA16(
                        output.data(), dtype, static_cast<size_t>(row * kOut + col));
                const double tolerance =
                        (dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3) *
                                std::abs(expected) +
                        1.0e-6;
                EXPECT_NEAR(actual, expected, tolerance)
                        << "row " << row << " col " << col;
            }
        }
    }
}

TEST(LinearA16Avx2Fma, BlockedProjectionNarrowsAfterAllKPanels) {
#if defined(A16_HAS_AVX2_FMA_KERNEL)
    const CpuBackend detected;
    const auto required = CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c});
    if (!detected.cpu_capabilities().effective_features.ContainsAll(required)) {
        GTEST_SKIP() << "AVX2/FMA/F16C are required to execute the blocked driver";
    }
    CpuBackend backend(Synthetic(required));
    constexpr int64_t m = 49, k = 521, n = 131;
    constexpr std::array<int64_t, 2> input_shape{m, k};
    constexpr std::array<int64_t, 2> weight_shape{n, k};
    constexpr std::array<int64_t, 2> output_shape{m, n};
    constexpr std::array<int64_t, 2> input_strides{k + 3, 1};
    constexpr std::array<int64_t, 2> weight_strides{k + 5, 1};
    constexpr std::array<int64_t, 2> output_strides{n + 4, 1};
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        std::vector<std::byte> input(1 + 2 * m * input_strides[0]);
        std::vector<std::byte> weight(1 + 2 * n * weight_strides[0]);
        for (int64_t row = 0; row < m; ++row) {
            const int64_t offset = row * input_strides[0];
            WriteA16(input.data() + 1, 512.0F, dtype, offset);
            WriteA16(input.data() + 1, static_cast<float>(row % 7 + 2) * 0.25F, dtype, offset + 1);
            WriteA16(input.data() + 1, -512.0F, dtype, offset + 512);
            WriteA16(input.data() + 1, 1.0F, dtype, offset + 520);
        }
        for (int64_t col = 0; col < n; ++col) {
            const int64_t offset = col * weight_strides[0];
            WriteA16(weight.data() + 1, 512.0F, dtype, offset);
            WriteA16(weight.data() + 1, static_cast<float>(col % 5 + 2) * 0.5F, dtype, offset + 1);
            WriteA16(weight.data() + 1, 512.0F, dtype, offset + 512);
            WriteA16(weight.data() + 1, 0.125F, dtype, offset + 520);
        }
        const std::array<TensorView, 2> inputs{
                TensorView(input.data() + 1, dtype, input_shape, input_strides),
                TensorView(weight.data() + 1, dtype, weight_shape, weight_strides)};
        for (const DataType output_dtype: {dtype, DataType::Float32()}) {
            const auto kernel = backend.PrepareKernel(
                    OpType::kLinear, MakeSelector(dtype, output_dtype), MakeParams(dtype, output_dtype));
            ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
            const size_t output_bytes = static_cast<size_t>(output_dtype.nbytes());
            constexpr std::byte sentinel{0xff};
            std::vector<std::byte> output(1 + output_bytes * m * output_strides[0], sentinel);
            const std::array<MutableTensorView, 1> outputs{
                    MutableTensorView(output.data() + 1, output_dtype, output_shape, output_strides)};
            PreparedParams prepared;
            ASSERT_TRUE(kernel->params_builder(
                                      {.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs}, prepared.storage.data())
                                .ok());
            ASSERT_TRUE(kernel->fn({.kernel_params = prepared.storage.data(), .attrs = kernel->attrs}).ok());

            // The first panel contains a sum above the F16 range and too large
            // for BF16 to retain the small term. The next panel cancels it.
            EXPECT_EQ(output.front(), sentinel);
            for (int64_t row = 0; row < m; ++row) {
                for (int64_t col = 0; col < n; ++col) {
                    const float expected = static_cast<float>((row % 7 + 2) * (col % 5 + 2) + 1) * 0.125F;
                    const size_t index = static_cast<size_t>(row * output_strides[0] + col);
                    float actual;
                    if (output_dtype == DataType::Float32()) {
                        std::memcpy(&actual, output.data() + 1 + index * output_bytes, output_bytes);
                    } else {
                        actual = ReadA16(output.data() + 1, dtype, index);
                    }
                    EXPECT_EQ(actual, expected) << "row=" << row << " col=" << col;
                }
                const size_t start = 1 + static_cast<size_t>(row * output_strides[0] + n) * output_bytes;
                for (size_t byte = 0; byte < 4 * output_bytes; ++byte) {
                    EXPECT_EQ(output[start + byte], sentinel);
                }
            }
        }
    }
#else
    GTEST_SKIP() << "A16 SIMD was not compiled";
#endif
}

TEST(LinearA16Avx2Fma, BlockedProjectionPreservesBf16ProductOverflow) {
#if defined(A16_HAS_AVX2_FMA_KERNEL)
    const CpuBackend backend;
    if (!backend.cpu_capabilities().effective_features.ContainsAll(
                CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c}))) {
        GTEST_SKIP() << "AVX2/FMA/F16C are required to execute the blocked driver";
    }
    const DataType dtype = DataType::BFloat(16);
    constexpr int64_t m = 9, k = 17, n = 131;
    std::array<std::byte, m * k * sizeof(uint16_t)> input{};
    std::array<std::byte, n * k * sizeof(uint16_t)> weight{};
    std::array<float, m * n> output{};
    output.fill(std::numeric_limits<float>::quiet_NaN());
    for (int64_t row = 0; row < m; ++row) {
        WriteA16(input.data(), -0x1p127F, dtype, row * k);
        WriteA16(input.data(), 0x1p127F, dtype, row * k + 8);
    }
    // A product in the final N block overflows before cancellation. A fused
    // evaluation would be finite, so the blocked path must use the reference.
    WriteA16(weight.data(), 1.0F, dtype, (n - 1) * k);
    WriteA16(weight.data(), 2.0F, dtype, (n - 1) * k + 8);
    const cpu::detail::GemmA16W16Args args{
            .gemm = {
                    .lhs = input.data(),
                    .rhs = weight.data(),
                    .output = reinterpret_cast<std::byte*>(output.data()),
                    .m = m,
                    .n = n,
                    .k = k,
                    .lhs_m_stride = k,
                    .lhs_k_stride = 1,
                    .rhs_k_stride = 1,
                    .rhs_n_stride = k,
                    .output_m_stride = n,
                    .output_n_stride = 1,
            },
            .operand_dtype = dtype,
            .output_dtype = DataType::Float32(),
    };
    ASSERT_TRUE(cpu::detail::RunGemmA16W16Avx2Fma(args).ok());
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t col = 0; col < n - 1; ++col) {
            EXPECT_EQ(output[row * n + col], 0.0F);
        }
        EXPECT_EQ(output[row * n + n - 1], std::numeric_limits<float>::infinity());
    }
#else
    GTEST_SKIP() << "A16 SIMD was not compiled";
#endif
}

} // namespace
