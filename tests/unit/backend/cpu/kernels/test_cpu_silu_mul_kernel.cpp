#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/silu_mul/silu_mul_internal.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>
#include <vector>

namespace {

using namespace aethermind;
using cpu::detail::SiluMulA32KernelArgs;

KernelSelector MakeSelector(const DataType& dtype = DataType::Float32()) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = dtype,
            .output_dtype = dtype,
            .weight_dtype = dtype,
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareSiluMulKernel(const DataType& dtype = DataType::Float32(),
                                              CpuFeaturePolicy policy = {}) {
    CpuBackend backend(policy);
    return backend.PrepareKernel(OpType::kSiluMul, MakeSelector(dtype), OpParams{SiluMulParams{}});
}

struct SiluMulTestViews {
    TensorView gate{};
    TensorView up{};
    MutableTensorView output{};
};

struct PreparedKernelParams {
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
};

StatusOr<PreparedKernelParams> BuildSiluMulPreparedParams(
        const ResolvedKernel& kernel,
        const SiluMulTestViews& views) noexcept {
    PreparedKernelParams prepared;
    const std::array<TensorView, 2> inputs{views.gate, views.up};
    const std::array<MutableTensorView, 1> outputs{views.output};
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

Status RunSiluMul(const SiluMulTestViews& views, CpuFeaturePolicy policy = {}) noexcept {
    const auto kernel = PrepareSiluMulKernel(DataType::Float32(), policy);
    if (!kernel.ok()) {
        return kernel.status();
    }
    const auto prepared = BuildSiluMulPreparedParams(*kernel, views);
    if (!prepared.ok()) {
        return prepared.status();
    }
    return kernel->fn(KernelContext{
            .kernel_params = prepared->storage.data(),
            .attrs = kernel->attrs,
    });
}

float ExpectedSilu(float value) {
    if (value >= 0.0F) {
        return value / (1.0F + std::exp(-value));
    }
    const float exp_value = std::exp(value);
    return value * exp_value / (1.0F + exp_value);
}

float ExpectedSiluMul(float gate, float up) {
    return ExpectedSilu(gate) * up;
}

TEST(CPUKernelSiluMul, SelectsF32KernelUsingEffectiveFeatures) {
    const auto kernel = PrepareSiluMulKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();

    bool expect_avx2 = false;
#if defined(RMSNORM_HAS_AVX2_FMA_KERNEL)
    const CpuBackend backend;
    expect_avx2 = backend.cpu_capabilities().effective_features.ContainsAll(
            CpuFeatureSet::From({CpuFeature::kAvx2, CpuFeature::kFma}));
#endif
    EXPECT_EQ(kernel->name, expect_avx2 ? "cpu::silu_mul_f32_avx2_fma"
                                        : "cpu::silu_mul_f32_reference");
    const auto fallback = PrepareSiluMulKernel(
            DataType::Float32(),
            CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})});
    ASSERT_TRUE(fallback.ok());
    EXPECT_EQ(fallback->name, "cpu::silu_mul_f32_reference");
    EXPECT_EQ(kernel->op_type, OpType::kSiluMul);
    EXPECT_EQ(kernel->params_size, sizeof(SiluMulA32KernelArgs));
    EXPECT_NE(kernel->params_builder, nullptr);
    EXPECT_TRUE(kernel->attrs.empty());
    EXPECT_EQ(kernel->workspace_requirement.bytes, 0U);
}

TEST(CPUKernelSiluMul, FlatInPlaceVectorAndTailMeetIndependentF64Budget) {
    constexpr int64_t n = 65539;
    constexpr std::array<int64_t, 1> shape{n};
    constexpr std::array<int64_t, 1> strides{1};
    const auto dtype = DataType::Float32();
    std::vector<float> gate(n + 1), up(n + 1);
    for (int64_t i = 0; i < n; ++i) {
        gate[i] = -16.0F + static_cast<float>(i % 65537) * 0x1p-11F;
        up[i] = 0.125F + static_cast<float>(i % 61) * 0.1F;
        if (i % 2 == 1) {
            up[i] = -up[i];
        }
    }
    gate[23] = 16.0F;
    gate[n] = up[n] = 24.0F;
    const auto input_gate = gate;
    const auto input_up = up;
    for (float* output: {gate.data(), up.data()}) {
        ASSERT_TRUE(RunSiluMul({TensorView(gate.data(), dtype, shape, strides),
                                TensorView(up.data(), dtype, shape, strides),
                                MutableTensorView(output, dtype, shape, strides)})
                            .ok());
        double max_relative_error = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            const double x = input_gate[i];
            const double expected = x * input_up[i] / (1.0 + std::exp(-x));
            ASSERT_TRUE(std::isfinite(output[i]));
            if (expected == 0.0) {
                EXPECT_EQ(output[i], 0.0F);
            } else {
                max_relative_error = std::max(max_relative_error, std::abs(output[i] / expected - 1.0));
            }
        }
        EXPECT_LE(max_relative_error, 6 * std::numeric_limits<float>::epsilon());
        EXPECT_EQ(output[n], 24.0F);
        // Restore the changed input without invalidating the borrowed pointer.
        std::copy(input_gate.begin(), input_gate.end(), gate.begin());
        std::copy(input_up.begin(), input_up.end(), up.begin());
    }
}

TEST(CPUKernelSiluMul, MixedVectorAndExceptionalBlocksPreserveScalarClassification) {
    constexpr int64_t n = 65;
    constexpr std::array<int64_t, 1> shape{n};
    constexpr std::array<int64_t, 1> strides{1};
    const auto dtype = DataType::Float32();
    std::array<float, n> gate{}, up{}, actual{}, expected{};
    gate.fill(1.75F);
    up.fill(2.0F);
    gate[0] = -0.0F;
    up[1] = -0.0F;
    gate[8] = -std::numeric_limits<float>::infinity();
    gate[16] = std::numeric_limits<float>::quiet_NaN();
    up[24] = -std::numeric_limits<float>::infinity();
    gate[32] = -std::numeric_limits<float>::denorm_min();
    up[40] = std::numeric_limits<float>::max();
    gate[48] = -100.0F;
    gate[56] = 0x1p-33F;
    gate[64] = -16.000002F;
    const auto views = [&](auto& output) {
        return SiluMulTestViews{TensorView(gate.data(), dtype, shape, strides),
                                TensorView(up.data(), dtype, shape, strides),
                                MutableTensorView(output.data(), dtype, shape, strides)};
    };
    ASSERT_TRUE(RunSiluMul(views(expected), CpuFeaturePolicy{
                                                    .disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})})
                        .ok());
    ASSERT_TRUE(RunSiluMul(views(actual)).ok());
    for (int64_t i = 0; i < n; ++i) {
        if (std::isnan(expected[i])) {
            EXPECT_TRUE(std::isnan(actual[i]));
        } else if (i < 2 || i >= 8) {
            EXPECT_EQ(std::bit_cast<uint32_t>(actual[i]), std::bit_cast<uint32_t>(expected[i]));
        }
    }
}

TEST(CPUKernelSiluMul, DirectedRoundingMatchesScalar) {
    constexpr int64_t n = 64;
    constexpr std::array<int64_t, 1> shape{n};
    constexpr std::array<int64_t, 1> strides{1};
    const auto dtype = DataType::Float32();
    std::array<float, n> gate{}, up{}, actual{}, expected{};
    gate.fill(-0.42F);
    up.fill(0.83F);
    const auto views = [&](auto& output) {
        return SiluMulTestViews{TensorView(gate.data(), dtype, shape, strides),
                                TensorView(up.data(), dtype, shape, strides),
                                MutableTensorView(output.data(), dtype, shape, strides)};
    };
    const int original = std::fegetround();
    ASSERT_NE(original, -1);
    ASSERT_EQ(std::fesetround(FE_DOWNWARD), 0);
    const auto scalar = RunSiluMul(
            views(expected), CpuFeaturePolicy{.disabled_features = CpuFeatureSet::From({CpuFeature::kAvx2})});
    const auto selected = RunSiluMul(views(actual));
    ASSERT_EQ(std::fesetround(original), 0);
    ASSERT_TRUE(scalar.ok());
    ASSERT_TRUE(selected.ok());
    EXPECT_EQ(actual, expected);
}

TEST(CPUKernelSiluMul, Float16SelectorResolvesA16Reference) {
    const auto kernel = PrepareSiluMulKernel(DataType::Float(16));
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    EXPECT_EQ(std::string_view{kernel->name}, "cpu::silu_mul_f16_reference");
}

TEST(CPUKernelSiluMul, ContiguousValuesUseStableLargeMagnitudeFormula) {
    constexpr std::array gate{-100.0F, -2.0F, 0.0F, 2.0F, 100.0F};
    constexpr std::array up{1.0F, 2.0F, -3.0F, 4.0F, 5.0F};
    std::array<float, gate.size()> output{};
    constexpr int64_t shape[1] = {static_cast<int64_t>(gate.size())};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), shape, strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (size_t index = 0; index < gate.size(); ++index) {
        EXPECT_FLOAT_EQ(output[index], ExpectedSiluMul(gate[index], up[index]));
        EXPECT_TRUE(std::isfinite(output[index]));
    }
    EXPECT_LT(output.front(), 0.0F);
    EXPECT_NEAR(output[1], -0.47681168F, 1.0e-7F);
    EXPECT_NEAR(output[3], 7.04637671F, 1.0e-6F);
    EXPECT_NEAR(output.back(), 500.0F, 1.0e-4F);
}

TEST(CPUKernelSiluMul, DifferentRankBroadcastWithArbitraryStrides) {
    constexpr int64_t gate_shape[3] = {2, 1, 3};
    constexpr int64_t gate_strides[3] = {5, 37, 1};
    constexpr int64_t up_shape[2] = {4, 1};
    constexpr int64_t up_strides[2] = {2, 19};
    constexpr int64_t output_shape[3] = {2, 4, 3};
    constexpr int64_t output_strides[3] = {1, 6, 2};
    constexpr float sentinel = -1234.0F;
    const std::array gate{-2.0F, -1.0F, 0.0F, sentinel, sentinel, 1.0F, 2.0F, 3.0F};
    const std::array up{1.0F, sentinel, 2.0F, sentinel, -3.0F, sentinel, 4.0F};
    std::array<float, 24> output{};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), gate_shape, gate_strides},
            .up = TensorView{up.data(), DataType::Float32(), up_shape, up_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), output_shape, output_strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (int64_t batch = 0; batch < output_shape[0]; ++batch) {
        for (int64_t row = 0; row < output_shape[1]; ++row) {
            for (int64_t column = 0; column < output_shape[2]; ++column) {
                const int64_t gate_offset = batch * gate_strides[0] + column * gate_strides[2];
                const int64_t up_offset = row * up_strides[0];
                const int64_t output_offset = batch * output_strides[0] +
                                              row * output_strides[1] +
                                              column * output_strides[2];
                EXPECT_FLOAT_EQ(output[static_cast<size_t>(output_offset)],
                                ExpectedSiluMul(gate[static_cast<size_t>(gate_offset)],
                                                up[static_cast<size_t>(up_offset)]));
            }
        }
    }
}

TEST(CPUKernelSiluMul, RankZeroGateBroadcastsAcrossVector) {
    const float gate = 2.0F;
    constexpr std::array up{1.0F, -2.0F, 3.0F};
    std::array<float, up.size()> output{};
    constexpr int64_t shape[1] = {static_cast<int64_t>(up.size())};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{&gate, DataType::Float32(), {}, {}},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (size_t index = 0; index < up.size(); ++index) {
        EXPECT_FLOAT_EQ(output[index], ExpectedSiluMul(gate, up[index]));
    }
}

TEST(CPUKernelSiluMul, ZeroElementBroadcastOutputNeedsNoGateOrOutputStorage) {
    constexpr int64_t gate_shape[2] = {0, 3};
    constexpr int64_t gate_strides[2] = {3, 1};
    constexpr int64_t up_shape[2] = {1, 3};
    constexpr int64_t up_strides[2] = {3, 1};
    constexpr std::array up{1.0F, 2.0F, 3.0F};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{nullptr, DataType::Float32(), gate_shape, gate_strides},
            .up = TensorView{up.data(), DataType::Float32(), up_shape, up_strides},
            .output = MutableTensorView{nullptr, DataType::Float32(), gate_shape, gate_strides},
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(CPUKernelSiluMul, AcceptsExactInPlaceAgainstGate) {
    std::array<float, 4> gate{-2.0F, -1.0F, 1.0F, 2.0F};
    constexpr std::array up{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> original = gate;
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), shape, strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{gate.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (size_t index = 0; index < gate.size(); ++index) {
        EXPECT_FLOAT_EQ(gate[index], ExpectedSiluMul(original[index], up[index]));
    }
}

TEST(CPUKernelSiluMul, AcceptsExactInPlaceAgainstUp) {
    constexpr std::array gate{-2.0F, -1.0F, 1.0F, 2.0F};
    std::array<float, 4> up{1.0F, 2.0F, 3.0F, 4.0F};
    const std::array<float, 4> original = up;
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), shape, strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{up.data(), DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    for (size_t index = 0; index < up.size(); ++index) {
        EXPECT_FLOAT_EQ(up[index], ExpectedSiluMul(gate[index], original[index]));
    }
}

TEST(CPUKernelSiluMul, RejectsBroadcastGateInsideOutputRegion) {
    std::array<float, 4> storage{2.0F, 0.0F, 0.0F, 0.0F};
    constexpr std::array up{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t scalar_shape[1] = {1};
    constexpr int64_t scalar_strides[1] = {1};
    constexpr int64_t output_shape[1] = {4};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{storage.data(), DataType::Float32(), scalar_shape, scalar_strides},
            .up = TensorView{up.data(), DataType::Float32(), output_shape, output_strides},
            .output = MutableTensorView{storage.data(), DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSiluMul, RejectsBroadcastUpInsideOutputRegion) {
    constexpr std::array gate{-2.0F, -1.0F, 1.0F, 2.0F};
    std::array<float, 4> storage{3.0F, 0.0F, 0.0F, 0.0F};
    constexpr int64_t scalar_shape[1] = {1};
    constexpr int64_t scalar_strides[1] = {1};
    constexpr int64_t output_shape[1] = {4};
    constexpr int64_t output_strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), output_shape, output_strides},
            .up = TensorView{storage.data(), DataType::Float32(), scalar_shape, scalar_strides},
            .output = MutableTensorView{storage.data(), DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSiluMul, RejectsPartialGateOutputOverlap) {
    std::array<float, 5> storage{1.0F, 2.0F, 3.0F, 4.0F, 5.0F};
    constexpr std::array up{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{storage.data(), DataType::Float32(), shape, strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{storage.data() + 1, DataType::Float32(), shape, strides},
    });
    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSiluMul, RejectsNonInjectiveOutputMapping) {
    constexpr std::array gate{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr std::array up{4.0F, 3.0F, 2.0F, 1.0F};
    std::array<float, 4> output{};
    constexpr int64_t shape[2] = {2, 2};
    constexpr int64_t input_strides[2] = {2, 1};
    constexpr int64_t output_strides[2] = {1, 1};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), shape, input_strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, input_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, output_strides},
    });
    EXPECT_EQ(status, StatusCode::kInvalidArgument) << status.ToString();
}

TEST(CPUKernelSiluMul, RejectsOutputLayoutWhoseInjectivityCannotBeProven) {
    constexpr std::array gate{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    constexpr std::array up{6.0F, 5.0F, 4.0F, 3.0F, 2.0F, 1.0F};
    std::array<float, 8> output{};
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t input_strides[2] = {3, 1};
    constexpr int64_t output_strides[2] = {3, 2};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate.data(), DataType::Float32(), shape, input_strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, input_strides},
            .output = MutableTensorView{output.data(), DataType::Float32(), shape, output_strides},
    });
    EXPECT_EQ(status, StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelSiluMul, ReportsUndecidableStridedOutputInputOverlap) {
    std::array<float, 9> storage{};
    constexpr std::array up{1.0F, 2.0F, 3.0F, 4.0F};
    constexpr int64_t shape[1] = {4};
    constexpr int64_t strides[1] = {2};

    const Status status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{storage.data(), DataType::Float32(), shape, strides},
            .up = TensorView{up.data(), DataType::Float32(), shape, strides},
            .output = MutableTensorView{storage.data() + 1, DataType::Float32(), shape, strides},
    });
    EXPECT_EQ(status, StatusCode::kUnimplemented) << status.ToString();
}

TEST(CPUKernelSiluMul, RejectsIncompatibleBroadcastDtypesAndAddressOverflow) {
    constexpr float gate[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    constexpr float up[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float output[4] = {};
    constexpr int64_t gate_shape[2] = {2, 2};
    constexpr int64_t up_shape[2] = {2, 3};
    constexpr int64_t output_shape[2] = {2, 2};
    constexpr int64_t gate_strides[2] = {2, 1};
    constexpr int64_t up_strides[2] = {3, 1};

    const Status broadcast_status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate, DataType::Float32(), gate_shape, gate_strides},
            .up = TensorView{up, DataType::Float32(), up_shape, up_strides},
            .output = MutableTensorView{output, DataType::Float32(), output_shape, gate_strides},
    });
    EXPECT_EQ(broadcast_status, StatusCode::kInvalidArgument) << broadcast_status.ToString();

    const Status dtype_status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate, DataType::Float32(), gate_shape, gate_strides},
            .up = TensorView{gate, DataType::Float32(), gate_shape, gate_strides},
            .output = MutableTensorView{output, DataType::Float(16), gate_shape, gate_strides},
    });
    EXPECT_EQ(dtype_status, StatusCode::kInvalidArgument) << dtype_status.ToString();

    constexpr int64_t huge_strides[2] = {std::numeric_limits<int64_t>::max(), 1};
    const Status overflow_status = RunSiluMul(SiluMulTestViews{
            .gate = TensorView{gate, DataType::Float32(), gate_shape, huge_strides},
            .up = TensorView{gate, DataType::Float32(), gate_shape, huge_strides},
            .output = MutableTensorView{output, DataType::Float32(), gate_shape, huge_strides},
    });
    EXPECT_EQ(overflow_status, StatusCode::kInvalidArgument) << overflow_status.ToString();
}

TEST(CPUKernelSiluMul, ParamsBuilderRejectsWrongArityAndInvalidViews) {
    const auto kernel = PrepareSiluMulKernel();
    ASSERT_TRUE(kernel.ok()) << kernel.status().ToString();
    PreparedKernelParams prepared;
    const std::array<TensorView, 1> one_input{TensorView{}};
    const std::array<MutableTensorView, 1> invalid_output{MutableTensorView{}};

    const Status arity_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = one_input,
                    .outputs = invalid_output,
                    .attrs = kernel->attrs,
            },
            prepared.storage.data());
    EXPECT_EQ(arity_status, StatusCode::kInvalidArgument) << arity_status.ToString();

    const std::array<TensorView, 2> invalid_inputs{TensorView{}, TensorView{}};
    const Status view_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = invalid_inputs,
                    .outputs = invalid_output,
                    .attrs = kernel->attrs,
            },
            prepared.storage.data());
    EXPECT_EQ(view_status, StatusCode::kInvalidArgument) << view_status.ToString();
}

} // namespace
