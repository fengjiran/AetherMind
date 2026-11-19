#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cstring>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;

StatusOr<ResolvedKernel> ResolveCast(DataType from, DataType to, DataType parameter) {
    const CpuBackend cpu;
    return cpu.PrepareKernel(OpType::kCast,
                             {.device_type = DeviceType::kCPU,
                              .act_dtype = from,
                              .output_dtype = to,
                              .weight_dtype = from,
                              .weight_format = WeightFormat::kPlain,
                              .phase = ExecPhase::kDecode},
                             CastParams{.target_dtype = parameter});
}

Status Cast(const TensorView& input, const MutableTensorView& output) {
    AM_ASSIGN_OR_RETURN(const auto kernel, ResolveCast(input.dtype(), output.dtype(), output.dtype()));
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
    const std::array inputs{input};
    const std::array outputs{output};
    AM_RETURN_IF_ERROR(kernel.params_builder({.inputs = inputs, .outputs = outputs, .attrs = kernel.attrs},
                                             storage.data()));
    return kernel.fn({.kernel_params = storage.data(), .attrs = kernel.attrs});
}

TEST(CPUKernelCast, ByteAlignedFp16RoundTripFollowsFixedRne) {
    constexpr std::array<uint32_t, 7> bits{
            0x80000000, 0x3f801000, 0x3f803000, 0x33800000, 0x33000000, 0x477ff000, 0x7f804000};
    constexpr std::array<uint16_t, 7> half{0x8000, 0x3c00, 0x3c02, 0x0001, 0x0000, 0x7c00, 0x7e02};
    constexpr std::array<uint32_t, 7> widened{
            0x80000000, 0x3f800000, 0x3f804000, 0x33800000, 0x00000000, 0x7f800000, 0x7fc04000};
    constexpr std::array<int64_t, 1> shape{static_cast<int64_t>(bits.size())};
    constexpr std::array<int64_t, 1> strides{1};
    std::array<std::byte, sizeof(bits) + 2> source{}, output{};
    std::array<std::byte, sizeof(half) + 2> narrowed{};
    std::memcpy(source.data() + 1, bits.data(), sizeof(bits));
    const auto f32 = DataType::Float32();
    const auto f16 = DataType::Float(16);
    ASSERT_TRUE(Cast(TensorView(source.data() + 1, f32, shape, strides),
                     MutableTensorView(narrowed.data() + 1, f16, shape, strides))
                        .ok());
    EXPECT_EQ(std::memcmp(narrowed.data() + 1, half.data(), sizeof(half)), 0);
    ASSERT_TRUE(Cast(TensorView(narrowed.data() + 1, f16, shape, strides),
                     MutableTensorView(output.data() + 1, f32, shape, strides))
                        .ok());
    EXPECT_EQ(std::memcmp(output.data() + 1, widened.data(), sizeof(widened)), 0);
    EXPECT_EQ(output.back(), std::byte{0});
}

TEST(CPUKernelCast, StridedOutputLeavesHolesAndRejectsAliasBeforeWriting) {
    constexpr std::array<int64_t, 2> shape{2, 2};
    constexpr std::array<int64_t, 2> src_strides{3, 1}, dst_strides{4, 2};
    std::array<float, 5> values{1, 2, -99, 3, 4};
    std::array<uint16_t, 7> output{0, 9, 0, 9, 0, 9, 0};
    const auto f32 = DataType::Float32();
    const auto f16 = DataType::Float(16);
    const TensorView input(values.data(), f32, shape, src_strides);
    ASSERT_TRUE(Cast(input, MutableTensorView(output.data(), f16, shape, dst_strides)).ok());
    EXPECT_EQ(output, (std::array<uint16_t, 7>{0x3c00, 9, 0x4000, 9, 0x4200, 9, 0x4400}));
    const auto saved = values;
    EXPECT_EQ(Cast(input, MutableTensorView(values.data(), f16, shape, dst_strides)),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(saved, values);
    constexpr std::array<int64_t, 2> non_injective{1, 1};
    EXPECT_EQ(Cast(input, MutableTensorView(output.data(), f16, shape, non_injective)),
              StatusCode::kInvalidArgument);
}

TEST(CPUKernelCast, EmptyBindingAndTargetValidation) {
    constexpr std::array<int64_t, 2> shape{0, 4}, strides{4, 1};
    const auto f32 = DataType::Float32();
    const auto f16 = DataType::Float(16);
    EXPECT_TRUE(Cast(TensorView(nullptr, f32, shape, strides),
                     MutableTensorView(nullptr, f16, shape, strides))
                        .ok());
    const auto inconsistent = ResolveCast(f32, f16, f32);
    EXPECT_EQ(inconsistent.status(), StatusCode::kInvalidArgument);
    const auto unsupported = ResolveCast(f32, DataType::BFloat(16), DataType::BFloat(16));
    EXPECT_FALSE(unsupported.ok());
}

} // namespace
