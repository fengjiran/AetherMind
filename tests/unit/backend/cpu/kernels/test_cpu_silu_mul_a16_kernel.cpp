#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/operators/op_params.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string_view>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeSelector(DataType act_dtype) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = act_dtype,
            .output_dtype = act_dtype,
            .weight_dtype = act_dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
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

void ExpectSiluMulMatchesOracle(DataType dtype) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kSiluMul, MakeSelector(dtype), OpParams{SiluMulParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name},
              dtype == DataType::Float(16) ? "cpu::silu_mul_f16_reference"
                                           : "cpu::silu_mul_bf16_reference");

    constexpr std::array<float, 4> gate_values{-1.0F, 0.0F, 1.0F, 2.0F};
    constexpr std::array<float, 4> up_values{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t strides[2] = {2, 1};

    std::array<std::byte, 8> gate_bytes{};
    std::array<std::byte, 8> up_bytes{};
    for (size_t index = 0; index < gate_values.size(); ++index) {
        WriteA16(gate_bytes.data(), gate_values[index], dtype, index);
        WriteA16(up_bytes.data(), up_values[index], dtype, index);
    }
    std::array<std::byte, 8> output_bytes{};

    const std::array<TensorView, 2> inputs{
            TensorView(gate_bytes.data(), dtype, shape, strides),
            TensorView(up_bytes.data(), dtype, shape, strides)};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output_bytes.data(), dtype, shape, strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> prepared{};
    ASSERT_TRUE(kernel->params_builder(
                              KernelParamsBuildContext{.inputs = inputs,
                                                       .outputs = outputs,
                                                       .attrs = kernel->attrs},
                              prepared.data())
                        .ok());
    ASSERT_TRUE(kernel->fn({.kernel_params = prepared.data(), .attrs = kernel->attrs}).ok());

    const double tolerance = dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3;
    for (size_t index = 0; index < gate_values.size(); ++index) {
        const double gate = static_cast<double>(gate_values[index]);
        const double expected = gate / (1.0 + std::exp(-gate)) *
                                static_cast<double>(up_values[index]);
        const double actual = ReadA16(output_bytes.data(), dtype, index);
        EXPECT_NEAR(actual, expected, tolerance * std::abs(expected) + 1.0e-7)
                << "index " << index;
    }
}

TEST(CPUKernelSiluMulA16, Float16MatchesIndependentOracle) {
    ExpectSiluMulMatchesOracle(DataType::Float(16));
}

TEST(CPUKernelSiluMulA16, BFloat16MatchesIndependentOracle) {
    ExpectSiluMulMatchesOracle(DataType::BFloat(16));
}

TEST(CPUKernelSiluMulA16, RejectsMixedOperandDtypes) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kSiluMul, MakeSelector(DataType::BFloat(16)), OpParams{SiluMulParams{}});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    constexpr int64_t shape[1] = {1};
    constexpr int64_t strides[1] = {1};
    std::array<std::byte, 2> gate{};
    std::array<std::byte, 2> up{};
    std::array<std::byte, 2> output{};
    const std::array<TensorView, 2> inputs{
            TensorView(gate.data(), DataType::BFloat(16), shape, strides),
            TensorView(up.data(), DataType::Float(16), shape, strides)};
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView(output.data(), DataType::BFloat(16), shape, strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> prepared{};
    EXPECT_EQ(kernel->params_builder(
                      KernelParamsBuildContext{.inputs = inputs,
                                               .outputs = outputs,
                                               .attrs = kernel->attrs},
                      prepared.data()),
              StatusCode::kInvalidArgument);
}

} // namespace
