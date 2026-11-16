#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"
#include "test_utils/test_allocator.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeSelector(DataType act_dtype, WeightFormat format) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = act_dtype,
            .weight_dtype = act_dtype,
            .weight_format = format,
            .phase = ExecPhase::kBoth,
    };
}

struct PreparedParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

Status RunEmbedding(const ResolvedKernel& kernel,
                    const TensorView& tokens,
                    const TensorView& weight,
                    const MutableTensorView& output) noexcept {
    PreparedParams prepared;
    const std::array<TensorView, 2> inputs{tokens, weight};
    const std::array<MutableTensorView, 1> outputs{output};
    AM_RETURN_IF_ERROR(kernel.params_builder(
            KernelParamsBuildContext{.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs},
            prepared.storage.data()));
    return kernel.fn(KernelContext{.kernel_params = prepared.storage.data(),
                                   .attrs = kernel.attrs});
}

void WriteBits(std::byte* destination, uint16_t bits, size_t index) noexcept {
    std::memcpy(destination + index * sizeof(bits), &bits, sizeof(bits));
}

uint16_t ReadBits(const std::byte* source, size_t index) noexcept {
    uint16_t bits = 0;
    std::memcpy(&bits, source + index * sizeof(bits), sizeof(bits));
    return bits;
}

TEST(CPUKernelEmbeddingA16, ResolvesPlainReferencesForBothDtypes) {
    CpuBackend backend;
    for (const DataType dtype: {DataType::Float(16), DataType::BFloat(16)}) {
        const auto kernel = backend.PrepareKernel(
                OpType::kEmbedding, MakeSelector(dtype, WeightFormat::kPlain),
                OpParams{EmbeddingParams{}});
        ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
        EXPECT_EQ(std::string_view{kernel->name},
                  dtype == DataType::Float(16) ? "cpu::embedding_a16_f16_reference"
                                               : "cpu::embedding_a16_bf16_reference");
    }
}

TEST(CPUKernelEmbeddingA16, Bf16GatherCopiesStoredBitsWithoutRounding) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kEmbedding,
            MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
            OpParams{EmbeddingParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    // Packed artifact-free gather: output rows must be bit-identical copies.
    constexpr int64_t weight_shape[2] = {3, 2};
    constexpr int64_t weight_strides[2] = {2, 1};
    constexpr int64_t token_ids[2] = {2, 0};
    constexpr int64_t token_shape[1] = {2};
    constexpr int64_t token_strides[1] = {1};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    constexpr std::array<uint16_t, 6> patterns{
            0x3F80, 0x4000, // row 0
            0x4040, 0x4080, // row 1
            0x3F00, 0x0001, // row 2
    };

    std::array<std::byte, 12> weight_bytes{};
    for (size_t index = 0; index < patterns.size(); ++index) {
        WriteBits(weight_bytes.data(), patterns[index], index);
    }
    std::array<std::byte, 8> output_bytes{};

    const TensorView weight(weight_bytes.data(), DataType::BFloat(16),
                            weight_shape, weight_strides);
    const TensorView tokens(token_ids, DataType::Int(64), token_shape, token_strides);
    const MutableTensorView output(output_bytes.data(), DataType::BFloat(16),
                                   output_shape, output_strides);
    ASSERT_TRUE(RunEmbedding(*kernel, tokens, weight, output).ok());

    EXPECT_EQ(ReadBits(output_bytes.data(), 0), patterns[4]);
    EXPECT_EQ(ReadBits(output_bytes.data(), 1), patterns[5]);
    EXPECT_EQ(ReadBits(output_bytes.data(), 2), patterns[0]);
    EXPECT_EQ(ReadBits(output_bytes.data(), 3), patterns[1]);
}

TEST(CPUKernelEmbeddingA16, RejectsFp32WeightUnderA16Selector) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kEmbedding,
            MakeSelector(DataType::BFloat(16), WeightFormat::kPlain),
            OpParams{EmbeddingParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr int64_t weight_shape[2] = {1, 2};
    constexpr int64_t weight_strides[2] = {2, 1};
    constexpr int64_t token_ids[1] = {0};
    constexpr int64_t token_shape[1] = {1};
    constexpr int64_t token_strides[1] = {1};
    constexpr int64_t output_shape[2] = {1, 2};
    constexpr int64_t output_strides[2] = {2, 1};
    std::array<float, 2> weight{1.0F, 2.0F};
    std::array<std::byte, 4> output{};

    EXPECT_EQ(RunEmbedding(*kernel,
                           TensorView(token_ids, DataType::Int(64), token_shape, token_strides),
                           TensorView(weight.data(), DataType::Float32(),
                                      weight_shape, weight_strides),
                           MutableTensorView(output.data(), DataType::BFloat(16),
                                             output_shape, output_strides)),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelEmbeddingA16, PackedIdentityGatherCopiesStoredBits) {
    constexpr int64_t weight_shape[2] = {2, 3};
    constexpr int64_t weight_strides[2] = {3, 1};
    constexpr std::array<uint16_t, 6> patterns{
            0x3F80,
            0x4000,
            0x4040,
            0x4080,
            0x3F00,
            0x0001,
    };
    std::array<std::byte, 12> weight_bytes{};
    for (size_t index = 0; index < patterns.size(); ++index) {
        WriteBits(weight_bytes.data(), patterns[index], index);
    }

    const KernelSelector selector =
            MakeSelector(DataType::BFloat(16), WeightFormat::kPacked);
    CpuBackend backend(CpuCapabilities{});
    const auto recipe = backend.GetPackingRecipe(OpType::kEmbedding, selector);
    ASSERT_TRUE(recipe.ok()) << recipe.status().ToString();
    const TensorView weight_view(weight_bytes.data(), DataType::BFloat(16),
                                 weight_shape, weight_strides);
    auto packed = backend.PackWeights(
            OpType::kEmbedding, std::array<TensorView, 1>{weight_view},
            selector, *recipe, test::TestCpuAllocator());
    ASSERT_TRUE(packed.ok()) << packed.status().ToString();
    ASSERT_EQ((*packed)->storage().nbytes(), weight_bytes.size());

    const auto kernel = backend.PrepareKernel(
            OpType::kEmbedding, selector, OpParams{EmbeddingParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name},
              "cpu::embedding_a16_bf16_packed_identity_reference");

    constexpr int64_t token_ids[1] = {1};
    constexpr int64_t token_shape[1] = {1};
    constexpr int64_t token_strides[1] = {1};
    constexpr int64_t output_shape[2] = {1, 3};
    constexpr int64_t output_strides[2] = {3, 1};
    std::array<std::byte, 6> output_bytes{};
    const MutableTensorView output(output_bytes.data(), DataType::BFloat(16),
                                   output_shape, output_strides);
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
    ASSERT_TRUE(kernel->params_builder(
                              KernelParamsBuildContext{
                                      .inputs = std::array<TensorView, 1>{
                                              TensorView(token_ids, DataType::Int(64),
                                                         token_shape, token_strides)},
                                      .outputs = std::array<MutableTensorView, 1>{output},
                                      .attrs = kernel->attrs,
                                      .packed_weight = packed_view,
                              },
                              prepared.storage.data())
                        .ok());
    ASSERT_TRUE(kernel->fn({.kernel_params = prepared.storage.data(),
                            .attrs = kernel->attrs})
                        .ok());
    EXPECT_EQ(ReadBits(output_bytes.data(), 0), patterns[3]);
    EXPECT_EQ(ReadBits(output_bytes.data(), 1), patterns[4]);
    EXPECT_EQ(ReadBits(output_bytes.data(), 2), patterns[5]);
}

} // namespace
