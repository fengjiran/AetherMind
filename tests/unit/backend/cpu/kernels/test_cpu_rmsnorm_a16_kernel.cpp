#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/cpu_capabilities.h"
#include "aethermind/backend/cpu/cpu_info.h"
#include "aethermind/backend/cpu/cpu_packed_weight_layout.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/test_fp_environment_helpers.h"
#include "test_utils/test_allocator.h"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using aethermind::test::ScopedNearestGradualUnderflow;

constexpr float kEps = 1.0e-5F;

KernelSelector MakeSelector(DataType act_dtype, WeightFormat format) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = act_dtype,
            .weight_dtype = DataType::Float32(),
            .weight_format = format,
            .phase = ExecPhase::kBoth,
    };
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunRmsNorm(const ResolvedKernel& kernel,
                  const TensorView& input,
                  const TensorView& weight,
                  const MutableTensorView& output) noexcept {
    PreparedParams prepared;
    const std::array<TensorView, 2> inputs{input, weight};
    const std::array<MutableTensorView, 1> outputs{output};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs},
            prepared.storage.data()));
    return kernel.fn(KernelContext{.kernel_params = prepared.storage.data(),
                                   .attrs = kernel.attrs});
}

void WriteA16(std::byte* destination, float value, DataType dtype, size_t index) noexcept {
    const uint16_t bits = dtype == DataType::Float(16) ? Half(value).bits() : BFloat16(value).x;
    std::memcpy(destination + index * sizeof(bits), &bits, sizeof(bits));
}

float ReadA16(const std::byte* source, DataType dtype, size_t index) noexcept {
    uint16_t bits = 0;
    std::memcpy(&bits, source + index * sizeof(bits), sizeof(bits));
    return dtype == DataType::Float(16) ? static_cast<float>(Half(bits, Half::from_bits()))
                                        : static_cast<float>(BFloat16(bits, BFloat16::from_bits()));
}

// Independent oracle: variance and rsqrt in double, then one narrowing. The
// tolerance below is half an ulp of the output dtype, so the oracle never needs
// the kernel's narrowing helper.
std::array<double, 4> OracleRmsNorm(const std::array<float, 4>& input,
                                    const std::array<float, 4>& weight) {
    double sum_sq = 0.0;
    for (const float value: input) {
        sum_sq += static_cast<double>(value) * static_cast<double>(value);
    }
    const double inv_rms = 1.0 / std::sqrt(sum_sq / 4.0 + static_cast<double>(kEps));
    std::array<double, 4> expected{};
    for (size_t index = 0; index < expected.size(); ++index) {
        expected[index] = static_cast<double>(input[index]) * inv_rms *
                          static_cast<double>(weight[index]);
    }
    return expected;
}

void ExpectRmsNormMatchesOracle(DataType dtype) {
    // Empty capabilities force the scalar reference so the name assertion and
    // the oracle comparison describe the reference contract.
    CpuBackend backend(CpuCapabilities{});
    const auto kernel = backend.PrepareKernel(
            OpType::kRmsNorm, MakeSelector(dtype, WeightFormat::kPlain),
            OpParams{RmsNormParams{.eps = kEps}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name},
              dtype == DataType::Float(16) ? "cpu::rmsnorm_a16_f16_reference"
                                           : "cpu::rmsnorm_a16_bf16_reference");

    constexpr std::array<float, 4> input_values{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr std::array<float, 4> weight_values{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t input_shape[1] = {4};
    constexpr int64_t input_strides[1] = {1};

    std::array<std::byte, 8> input_bytes{};
    for (size_t index = 0; index < input_values.size(); ++index) {
        WriteA16(input_bytes.data(), input_values[index], dtype, index);
    }
    std::array<std::byte, 8> output_bytes{};

    ASSERT_TRUE(RunRmsNorm(*kernel,
                           TensorView(input_bytes.data(), dtype, input_shape, input_strides),
                           TensorView(weight_values.data(), DataType::Float32(),
                                      input_shape, input_strides),
                           MutableTensorView(output_bytes.data(), dtype,
                                             input_shape, input_strides))
                        .ok());

    const double tolerance = dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3;
    const auto expected = OracleRmsNorm(input_values, weight_values);
    for (size_t index = 0; index < expected.size(); ++index) {
        const double actual = ReadA16(output_bytes.data(), dtype, index);
        EXPECT_NEAR(actual, expected[index], tolerance * std::abs(expected[index]))
                << "index " << index;
    }
}

TEST(CPUKernelRmsNormA16, Float16MatchesIndependentOracle) {
    ExpectRmsNormMatchesOracle(DataType::Float(16));
}

TEST(CPUKernelRmsNormA16, BFloat16MatchesIndependentOracle) {
    ExpectRmsNormMatchesOracle(DataType::BFloat(16));
}

TEST(CPUKernelRmsNormA16, BFloat16SubnormalOutputsAgreeAcrossVectorBodyAndTail) {
    const ScopedNearestGradualUnderflow environment;
    ASSERT_TRUE(environment.ok());
    const auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();

    constexpr int64_t kHidden = 17;
    constexpr std::array<int64_t, 1> shape{kHidden};
    constexpr std::array<int64_t, 1> strides{1};
    std::array<uint16_t, kHidden> input{};
    input.fill(0x3F80U); // one, so eps=3 makes inv_rms exactly one half
    std::array<float, kHidden> weight{};
    weight.fill(std::bit_cast<float>(uint32_t{0x00020000U})); // 2^-132

    for (const CpuCapabilities& capabilities: {CpuCapabilities{}, *detected}) {
        CpuBackend backend(capabilities);
        const auto kernel = backend.PrepareKernel(
                OpType::kRmsNorm, MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
                OpParams{RmsNormParams{.eps = 3.0F}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        SCOPED_TRACE(kernel->name);
        std::array<uint16_t, kHidden> output{};
        ASSERT_TRUE(RunRmsNorm(*kernel,
                               TensorView(input.data(), DataType::BFloat(16), shape, strides),
                               TensorView(weight.data(), DataType::Float32(), shape, strides),
                               MutableTensorView(output.data(), DataType::BFloat(16), shape, strides))
                            .ok());
        for (size_t index = 0; index < output.size(); ++index) {
            // 2^-133 is exactly the smallest positive BF16 subnormal. On an
            // AVX512-BF16 host this covers 16 vector lanes and the scalar tail.
            EXPECT_EQ(output[index], 0x0001U) << "index " << index;
        }
    }
}

TEST(CPUKernelRmsNormA16, BFloat16NarrowingPreservesSignedUnderflowAndNormalBoundary) {
    const ScopedNearestGradualUnderflow environment;
    ASSERT_TRUE(environment.ok());
    const auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();

    // Independent F32 input bits and BF16 RNE results. Every case appears in
    // both a vector body and its tail; neither expected value uses BFloat16.
    struct NarrowingCase {
        uint32_t scaled_bits;
        uint16_t expected_bits;
    };
    constexpr std::array cases{
            NarrowingCase{0x00000000U, 0x0000U},
            NarrowingCase{0x80000000U, 0x8000U},
            NarrowingCase{0x00008000U, 0x0000U},
            NarrowingCase{0x00008002U, 0x0001U},
            NarrowingCase{0x00010000U, 0x0001U},
            NarrowingCase{0x00018000U, 0x0002U},
            NarrowingCase{0x007F0000U, 0x007FU},
            NarrowingCase{0x007F8000U, 0x0080U},
            NarrowingCase{0x00800000U, 0x0080U},
            NarrowingCase{0x80008000U, 0x8000U},
            NarrowingCase{0x80010000U, 0x8001U},
            NarrowingCase{0x807F8000U, 0x8080U},
            NarrowingCase{0x3F808000U, 0x3F80U},
            NarrowingCase{0x3F818000U, 0x3F82U},
    };
    constexpr int64_t kHidden = 17;
    constexpr std::array<int64_t, 1> shape{kHidden};
    constexpr std::array<int64_t, 1> strides{1};
    std::array<uint16_t, kHidden> input{};
    input.fill(0x3F80U);

    for (const CpuCapabilities& capabilities: {CpuCapabilities{}, *detected}) {
        CpuBackend backend(capabilities);
        const auto kernel = backend.PrepareKernel(
                OpType::kRmsNorm, MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
                OpParams{RmsNormParams{.eps = 3.0F}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        SCOPED_TRACE(kernel->name);
        for (const auto& test_case: cases) {
            SCOPED_TRACE(test_case.scaled_bits);
            std::array<float, kHidden> weight{};
            weight.fill(std::bit_cast<float>(test_case.scaled_bits) * 2.0F);
            std::array<uint16_t, kHidden> output{};
            ASSERT_TRUE(RunRmsNorm(*kernel,
                                   TensorView(input.data(), DataType::BFloat(16), shape, strides),
                                   TensorView(weight.data(), DataType::Float32(), shape, strides),
                                   MutableTensorView(output.data(), DataType::BFloat(16), shape, strides))
                                .ok());
            for (size_t index = 0; index < output.size(); ++index) {
                EXPECT_EQ(output[index], test_case.expected_bits) << "index " << index;
            }
        }
    }
}

TEST(CPUKernelRmsNormA16, KeepsFloat32WeightUnderA16Activations) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kRmsNorm,
            MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
            OpParams{RmsNormParams{.eps = kEps}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr int64_t shape[1] = {2};
    constexpr int64_t strides[1] = {1};
    std::array<std::byte, 4> input{};
    std::array<std::byte, 4> output{};
    std::array<std::byte, 4> half_weight{};

    EXPECT_EQ(RunRmsNorm(*kernel,
                         TensorView(input.data(), DataType::BFloat(16), shape, strides),
                         TensorView(half_weight.data(), DataType::Float(16), shape, strides),
                         MutableTensorView(output.data(), DataType::BFloat(16), shape, strides)),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelRmsNormA16, RejectsUnalignedFloat32Weight) {
    CpuBackend backend(CpuCapabilities{});
    const auto kernel = backend.PrepareKernel(
            OpType::kRmsNorm,
            MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
            OpParams{RmsNormParams{.eps = kEps}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr int64_t shape[1] = {2};
    constexpr int64_t strides[1] = {1};
    std::array<std::byte, 4> input{};
    std::array<std::byte, 4> output{};
    alignas(float) std::array<std::byte, 2 * sizeof(float) + alignof(float)> weight_storage{};
    const void* const misaligned_weight = weight_storage.data() + 1;
    ASSERT_NE(reinterpret_cast<std::uintptr_t>(misaligned_weight) % alignof(float), 0U);

    // Unspecified view alignment passes TensorView validity; only the entry's
    // natural alignment check may reject the misaligned float weight.
    const TensorView misaligned_view(misaligned_weight, DataType::Float32(), shape, strides);
    ASSERT_TRUE(misaligned_view.is_valid());

    PreparedParams prepared;
    const std::array<TensorView, 2> inputs{
            TensorView(input.data(), DataType::BFloat(16), shape, strides), misaligned_view};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::BFloat(16), shape, strides)};
    EXPECT_EQ(kernel->params_builder(
                      KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel->attrs},
                      prepared.storage.data()),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelRmsNormA16, PackedIdentityRejectsMisalignedStorageBeforePreparingParams) {
    constexpr size_t kAlignment = cpu::kCpuIdentityPackingAlignment;
    constexpr std::array<int64_t, 1> shape{2};
    constexpr std::array<int64_t, 1> strides{1};
    constexpr std::array<float, 2> weights{1.0F, 0.5F};
    CpuBackend reference_backend(CpuCapabilities{});
    CpuBackend host_backend;

    for (CpuBackend* backend: {&reference_backend, &host_backend}) {
        for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
            const auto kernel = backend->PrepareKernel(
                    OpType::kRmsNorm, MakeSelector(dtype, WeightFormat::kPacked),
                    OpParams{RmsNormParams{.eps = kEps}});
            ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
            SCOPED_TRACE(kernel->name);

            std::array<std::byte, 4> input{};
            WriteA16(input.data(), 2.0F, dtype, 0);
            WriteA16(input.data(), 4.0F, dtype, 1);
            std::array<std::byte, 4> output{};
            output.fill(std::byte{0x5A});
            const auto original_output = output;
            alignas(kAlignment) std::array<std::byte, 2 * kAlignment> weight_storage{};
            std::memcpy(weight_storage.data(), weights.data(), sizeof(weights));
            const std::array<TensorView, 1> inputs{
                    TensorView(input.data(), dtype, shape, strides)};
            const std::array<MutableTensorView, 1> outputs{
                    MutableTensorView(output.data(), dtype, shape, strides)};
            const PackedWeightView aligned_weight{
                    .data = weight_storage.data(),
                    .nbytes = sizeof(weights),
                    .logical_dtype = DataType::Float32(),
                    .logical_shape = shape,
                    .recipe_layout = cpu::kCpuIdentityPackingLayout,
                    .recipe_alignment = kAlignment,
                    .alignment = kAlignment,
            };
            PreparedParams prepared;
            ASSERT_TRUE(kernel->params_builder(
                                      KernelParamsBuildContext{
                                              .inputs = inputs,
                                              .outputs = outputs,
                                              .attrs = kernel->attrs,
                                              .packed_weight = aligned_weight},
                                      prepared.storage.data())
                                .ok());
            EXPECT_EQ(output, original_output);

            // Check both a naturally unaligned float and a float-aligned
            // address that still violates the packed recipe's 64-byte boundary.
            for (const size_t offset: {size_t{1}, alignof(float)}) {
                SCOPED_TRACE(offset);
                const auto* misaligned = weight_storage.data() + offset;
                ASSERT_NE(reinterpret_cast<std::uintptr_t>(misaligned) % kAlignment, 0U);
                if (offset == alignof(float)) {
                    ASSERT_EQ(reinterpret_cast<std::uintptr_t>(misaligned) % alignof(float), 0U);
                }
                std::memcpy(weight_storage.data() + offset, weights.data(), sizeof(weights));
                PackedWeightView weight = aligned_weight;
                weight.data = misaligned;
                prepared.storage.fill(std::byte{0xA5});
                const auto original_params = prepared.storage;

                EXPECT_EQ(kernel->params_builder(
                                  KernelParamsBuildContext{
                                          .inputs = inputs,
                                          .outputs = outputs,
                                          .attrs = kernel->attrs,
                                          .packed_weight = weight},
                                  prepared.storage.data()),
                          StatusCode::kInvalidArgument);
                EXPECT_EQ(prepared.storage, original_params);
                EXPECT_EQ(output, original_output);
            }
        }
    }
}

TEST(CPUKernelRmsNormA16, PackedIdentityConsumesFloat32Artifact) {
    constexpr int64_t shape[1] = {2};
    constexpr int64_t strides[1] = {1};
    constexpr std::array<float, 2> weight_values{1.0F, 0.5F};
    const KernelSelector selector =
            MakeSelector(DataType::BFloat(16), WeightFormat::kPacked);

    CpuBackend backend(CpuCapabilities{});
    const auto recipe = backend.GetPackingRecipe(OpType::kRmsNorm, selector);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    auto packed = backend.PackWeights(
            OpType::kRmsNorm,
            std::array<TensorView, 1>{TensorView(weight_values.data(), DataType::Float32(),
                                                 shape, strides)},
            selector, *recipe, test::TestCpuAllocator());
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_EQ((*packed)->storage().nbytes(), weight_values.size() * sizeof(float));

    const auto kernel = backend.PrepareKernel(
            OpType::kRmsNorm, selector, OpParams{RmsNormParams{.eps = kEps}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name},
              "cpu::rmsnorm_a16_bf16_packed_identity_reference");

    std::array<std::byte, 4> input{};
    WriteA16(input.data(), 2.0F, DataType::BFloat(16), 0);
    WriteA16(input.data(), 4.0F, DataType::BFloat(16), 1);
    std::array<std::byte, 4> output{};
    const PackedWeightView packed_view{
            .data = (*packed)->storage().data(),
            .nbytes = (*packed)->storage().nbytes(),
            .logical_dtype = (*packed)->logical_dtype(),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };

    PreparedParams prepared;
    const MutableTensorView output_view(output.data(), DataType::BFloat(16),
                                        shape, strides);
    ASSERT_TRUE(kernel->params_builder(
                              KernelParamsBuildContext{
                                      .inputs = std::array<TensorView, 1>{
                                              TensorView(input.data(), DataType::BFloat(16),
                                                         shape, strides)},
                                      .outputs = std::array<MutableTensorView, 1>{output_view},
                                      .attrs = kernel->attrs,
                                      .packed_weight = packed_view,
                              },
                              prepared.storage.data())
                        .ok());
    ASSERT_TRUE(kernel->fn({.kernel_params = prepared.storage.data(),
                            .attrs = kernel->attrs})
                        .ok());

    // sqrt(mean(4, 16)) = sqrt(10); outputs are x * inv_rms * w.
    const double inv_rms = 1.0 / std::sqrt(10.0 + static_cast<double>(kEps));
    EXPECT_NEAR(static_cast<double>(ReadA16(output.data(), DataType::BFloat(16), 0)),
                2.0 * inv_rms, 4.0e-3);
    EXPECT_NEAR(static_cast<double>(ReadA16(output.data(), DataType::BFloat(16), 1)),
                4.0 * inv_rms * 0.5, 4.0e-3);

    // A 16-bit artifact violates the retained-FP32 Norm contract.
    std::array<std::byte, 4> half_weight{};
    const PackedWeightView wrong_dtype{
            .data = half_weight.data(),
            .nbytes = half_weight.size(),
            .logical_dtype = DataType::BFloat(16),
            .logical_shape = (*packed)->logical_shape(),
            .recipe_layout = (*packed)->recipe().layout,
            .recipe_alignment = (*packed)->recipe().alignment,
            .alignment = (*packed)->storage().alignment(),
    };
    EXPECT_EQ(kernel->params_builder(
                      KernelParamsBuildContext{
                              .inputs = std::array<TensorView, 1>{
                                      TensorView(input.data(), DataType::BFloat(16),
                                                 shape, strides)},
                              .outputs = std::array<MutableTensorView, 1>{output_view},
                              .attrs = kernel->attrs,
                              .packed_weight = wrong_dtype,
                      },
                      prepared.storage.data()),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelRmsNormA16, Avx2FmaMatchesDoubleOracleWithTailElements) {
    const auto detected = cpu::DetectCpuCapabilities();
    ASSERT_TRUE(detected.ok()) << detected.status().ToString();
    const auto required = CpuFeatureSet::From(
            {CpuFeature::kAvx2, CpuFeature::kFma, CpuFeature::kF16c});
    if (!detected->effective_features.ContainsAll(required)) {
        GTEST_SKIP() << "AVX2/FMA/F16C are required to execute the fallback";
    }

    constexpr int64_t kHidden = 10; // eight-wide body plus a two-element tail
    constexpr std::array<float, kHidden> input_values{
            1.0F, -2.0F, 0.5F, 3.0F, -0.25F, 2.0F, 1.5F, -1.0F, 0.75F, -0.5F};
    constexpr std::array<float, kHidden> weight_values{
            1.0F, 0.5F, -1.0F, 2.0F, 0.25F, -0.5F, 1.5F, -1.5F, 0.75F, 1.0F};
    constexpr int64_t shape[1] = {kHidden};
    constexpr int64_t strides[1] = {1};

    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        CpuBackend backend(CpuCapabilities{
                .architecture = CpuArchitecture::kX86_64,
                .usable_features = required,
                .effective_features = required,
        });
        const auto kernel = backend.PrepareKernel(
                OpType::kRmsNorm, MakeSelector(dtype, WeightFormat::kPlain),
                OpParams{RmsNormParams{.eps = kEps}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        ASSERT_EQ(std::string_view{kernel->name},
                  dtype == DataType::Float(16) ? "cpu::rmsnorm_a16_f16_avx2_fma"
                                               : "cpu::rmsnorm_a16_bf16_avx2_fma");

        std::array<std::byte, kHidden * 2> input_bytes{};
        std::array<std::byte, kHidden * 2> output_bytes{};
        for (size_t index = 0; index < input_values.size(); ++index) {
            WriteA16(input_bytes.data(), input_values[index], dtype, index);
        }

        ASSERT_TRUE(RunRmsNorm(*kernel,
                               TensorView(input_bytes.data(), dtype, shape, strides),
                               TensorView(weight_values.data(), DataType::Float32(),
                                          shape, strides),
                               MutableTensorView(output_bytes.data(), dtype,
                                                 shape, strides))
                            .ok());

        double sum_sq = 0.0;
        for (size_t index = 0; index < input_values.size(); ++index) {
            const double widened = ReadA16(input_bytes.data(), dtype, index);
            sum_sq += widened * widened;
        }
        const double inv_rms = 1.0 / std::sqrt(sum_sq / static_cast<double>(kHidden) +
                                               static_cast<double>(kEps));
        const double tolerance = dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3;
        for (size_t index = 0; index < input_values.size(); ++index) {
            const double expected = ReadA16(input_bytes.data(), dtype, index) * inv_rms *
                                    static_cast<double>(weight_values[index]);
            EXPECT_NEAR(static_cast<double>(ReadA16(output_bytes.data(), dtype, index)),
                        expected, tolerance * std::abs(expected) + 1.0e-7)
                    << "index " << index;
        }
    }
}

} // namespace
