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
#include <vector>

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

constexpr int64_t kSeqLen = 2;
constexpr int64_t kHeadDim = 4;
constexpr int64_t kNumQHeads = 2;
constexpr int64_t kNumKvHeads = 1;
constexpr double kTheta = 4.0;

RoPEParams MakeParams() {
    return RoPEParams{
            .head_dim = kHeadDim,
            .num_q_heads = kNumQHeads,
            .num_kv_heads = kNumKvHeads,
            .max_pos_embeddings = 8,
            .theta = kTheta,
            .algorithm = StandardRoPE{},
    };
}

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

// Independent split-half rotation oracle in double precision. The kernel
// narrows exactly once, so comparing within half an output-dtype ulp keeps the
// oracle free of the kernel's conversion helpers.
std::vector<double> OracleRoPE(const std::vector<float>& input,
                               int64_t num_heads,
                               const std::array<int64_t, 2>& positions) {
    std::vector<double> output(static_cast<size_t>(kSeqLen * num_heads * kHeadDim), 0.0);
    const int64_t half = kHeadDim / 2;
    for (int64_t token = 0; token < kSeqLen; ++token) {
        for (int64_t pair = 0; pair < half; ++pair) {
            const double inv_freq =
                    std::pow(kTheta, -2.0 * static_cast<double>(pair) / static_cast<double>(kHeadDim));
            const double angle = static_cast<double>(positions[static_cast<size_t>(token)]) * inv_freq;
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            for (int64_t head = 0; head < num_heads; ++head) {
                const int64_t base = (token * num_heads + head) * kHeadDim;
                const double first = static_cast<double>(input[static_cast<size_t>(base + pair)]);
                const double second =
                        static_cast<double>(input[static_cast<size_t>(base + half + pair)]);
                output[static_cast<size_t>(base + pair)] = first * cosine - second * sine;
                output[static_cast<size_t>(base + half + pair)] = second * cosine + first * sine;
            }
        }
    }
    return output;
}

void ExpectRoPEMatchesOracle(DataType dtype) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kRoPE, MakeSelector(dtype), OpParams{MakeParams()});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name},
              dtype == DataType::Float(16) ? "cpu::rope_f16_reference"
                                           : "cpu::rope_bf16_reference");

    const std::array<int64_t, 2> positions{0, 1};
    constexpr std::array<float, 8> q_values{
            1.0F, 2.0F, 0.5F, -1.0F,  // token 0, head 0
            -2.0F, 1.0F, 0.25F, 0.5F, // token 0, head 1
    };
    constexpr std::array<float, 4> k_values{1.0F, -1.0F, 2.0F, 0.5F};

    const int64_t q_width = kNumQHeads * kHeadDim;
    const int64_t k_width = kNumKvHeads * kHeadDim;
    const std::array<int64_t, 2> q_shape{kSeqLen, q_width};
    const std::array<int64_t, 2> k_shape{kSeqLen, k_width};
    const std::array<int64_t, 2> q_strides{q_width, 1};
    const std::array<int64_t, 2> k_strides{k_width, 1};
    const std::array<int64_t, 1> pos_shape{kSeqLen};
    const std::array<int64_t, 1> pos_strides{1};

    std::vector<std::byte> q_bytes(static_cast<size_t>(kSeqLen * q_width) * 2);
    std::vector<std::byte> k_bytes(static_cast<size_t>(kSeqLen * k_width) * 2);
    for (size_t index = 0; index < q_bytes.size() / 2; ++index) {
        WriteA16(q_bytes.data(), q_values[index % q_values.size()], dtype, index);
    }
    for (size_t index = 0; index < k_bytes.size() / 2; ++index) {
        WriteA16(k_bytes.data(), k_values[index % k_values.size()], dtype, index);
    }
    std::vector<std::byte> q_output(q_bytes.size());
    std::vector<std::byte> k_output(k_bytes.size());

    const std::array<TensorView, 3> inputs{
            TensorView(q_bytes.data(), dtype, q_shape, q_strides),
            TensorView(k_bytes.data(), dtype, k_shape, k_strides),
            TensorView(positions.data(), DataType::Int(64), pos_shape, pos_strides)};
    const std::array<MutableTensorView, 2> outputs{
            MutableTensorView(q_output.data(), dtype, q_shape, q_strides),
            MutableTensorView(k_output.data(), dtype, k_shape, k_strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> prepared{};
    ASSERT_TRUE(kernel->params_builder(
                              KernelParamsBuildContext{.inputs = inputs,
                                                       .outputs = outputs,
                                                       .attrs = kernel->attrs},
                              prepared.data())
                        .ok());
    ASSERT_TRUE(kernel->fn({.kernel_params = prepared.data(), .attrs = kernel->attrs}).ok());

    std::vector<float> q_input(static_cast<size_t>(kSeqLen * q_width));
    std::vector<float> k_input(static_cast<size_t>(kSeqLen * k_width));
    for (size_t index = 0; index < q_input.size(); ++index) {
        q_input[index] = ReadA16(q_bytes.data(), dtype, index);
    }
    for (size_t index = 0; index < k_input.size(); ++index) {
        k_input[index] = ReadA16(k_bytes.data(), dtype, index);
    }

    const double tolerance = dtype == DataType::Float(16) ? 4.9e-4 : 3.9e-3;
    const auto expected_q = OracleRoPE(q_input, kNumQHeads, positions);
    for (size_t index = 0; index < expected_q.size(); ++index) {
        const double actual = ReadA16(q_output.data(), dtype, index);
        EXPECT_NEAR(actual, expected_q[index], tolerance * std::abs(expected_q[index]) + 2.0e-7)
                << "q index " << index;
    }

    const auto expected_k = OracleRoPE(k_input, kNumKvHeads, positions);
    for (size_t index = 0; index < expected_k.size(); ++index) {
        const double actual = ReadA16(k_output.data(), dtype, index);
        EXPECT_NEAR(actual, expected_k[index], tolerance * std::abs(expected_k[index]) + 2.0e-7)
                << "k index " << index;
    }
}

TEST(CPUKernelRoPEA16, Float16MatchesIndependentOracle) {
    ExpectRoPEMatchesOracle(DataType::Float(16));
}

TEST(CPUKernelRoPEA16, BFloat16MatchesIndependentOracle) {
    ExpectRoPEMatchesOracle(DataType::BFloat(16));
}

TEST(CPUKernelRoPEA16, RejectsFloat32ViewsUnderA16Selector) {
    CpuBackend backend;
    const auto kernel = backend.PrepareKernel(
            OpType::kRoPE, MakeSelector(DataType::BFloat(16)), OpParams{MakeParams()});
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    const int64_t q_width = kNumQHeads * kHeadDim;
    const int64_t k_width = kNumKvHeads * kHeadDim;
    const std::array<int64_t, 2> q_shape{kSeqLen, q_width};
    const std::array<int64_t, 2> k_shape{kSeqLen, k_width};
    const std::array<int64_t, 2> q_strides{q_width, 1};
    const std::array<int64_t, 2> k_strides{k_width, 1};
    const std::array<int64_t, 1> pos_shape{kSeqLen};
    const std::array<int64_t, 1> pos_strides{1};
    std::vector<float> q(static_cast<size_t>(kSeqLen * q_width));
    std::vector<float> k(static_cast<size_t>(kSeqLen * k_width));
    std::vector<float> q_output(q.size());
    std::vector<float> k_output(k.size());
    const std::array<int64_t, 2> positions{0, 1};

    const std::array<TensorView, 3> inputs{
            TensorView(q.data(), DataType::Float32(), q_shape, q_strides),
            TensorView(k.data(), DataType::Float32(), k_shape, k_strides),
            TensorView(positions.data(), DataType::Int(64), pos_shape, pos_strides)};
    const std::array<MutableTensorView, 2> outputs{
            MutableTensorView(q_output.data(), DataType::Float32(), q_shape, q_strides),
            MutableTensorView(k_output.data(), DataType::Float32(), k_shape, k_strides)};
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> prepared{};
    EXPECT_EQ(kernel->params_builder(
                      KernelParamsBuildContext{.inputs = inputs,
                                               .outputs = outputs,
                                               .attrs = kernel->attrs},
                      prepared.data()),
              StatusCode::kInvalidArgument);
}

} // namespace
