#include "../test_graph_helpers.h"
#include "test_const_eval_helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

namespace {

using namespace aethermind;

using namespace test_utils;

TEST(ConstEvaluator, FindsSiluMulEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kSiluMul), nullptr);
}

// ── SiluMul rank-zero scalar tests ──

TEST(ConstEvaluator, PlansSiluMulRankZero) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "fused"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluMulParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, static_cast<size_t>(DataType::Float32().nbytes()));
    EXPECT_TRUE(plan->outputs[0].strides.empty());
}

template<typename T>
void ExpectSiluMulRankZeroEvaluation(DataType dtype, T gate_value, T up_value, T expected_value) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{};
    const std::vector<int64_t> strides{};
    const std::vector<std::byte> gate_bytes = BytesFromValues(std::vector<T>{gate_value});
    const std::vector<std::byte> up_bytes = BytesFromValues(std::vector<T>{up_value});
    std::vector<std::byte> output_bytes(sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), dtype, shape, strides),
            TensorView(up_bytes.data(), dtype, shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), 1U);
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_NEAR(result[0], expected_value, 1e-5F);
    } else {
        EXPECT_EQ(result[0], expected_value);
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulRankZeroFloat32) {
    float x = 2.0F;
    float silu;
    if (x >= 0.0F) {
        silu = x / (1.0F + std::exp(-x));
    } else {
        silu = x * std::exp(x) / (1.0F + std::exp(x));
    }
    float expected = silu * 3.0F;
    ExpectSiluMulRankZeroEvaluation<float>(DataType::Float32(), x, 3.0F, expected);
}

TEST(ConstEvaluator, EvaluatesSiluMulRankZeroBFloat16) {
    float x = 2.0F;
    float silu;
    if (x >= 0.0F) {
        silu = x / (1.0F + std::exp(-x));
    } else {
        silu = x * std::exp(x) / (1.0F + std::exp(x));
    }
    float expected = silu * 3.0F;
    ExpectSiluMulRankZeroEvaluation<BFloat16>(
            DataType::BFloat(16),
            BFloat16(2.0F),
            BFloat16(3.0F),
            BFloat16(expected));
}

// SiluMul rank-zero scalar-tensor mismatch — must still be rejected
TEST(ConstEvaluator, SkipsSiluMulRankZeroScalarTensorMismatch) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

// ── SiluMul dtype coverage — mirrors kSiluMulSupportedDTypes ──

TEST(ConstEvaluator, PlansSiluMulSupportedDTypesSameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<DataType> dtypes = {
            DataType::Float32(),
            DataType::Float(16),
            DataType::BFloat(16),
            DataType::Float8E4M3FN(),
            DataType::Float8E5M2(),
    };

    for (const DataType dtype: dtypes) {
        const TensorSpec spec = Spec(dtype, {2});
        const std::vector<GraphValueDesc> inputs = {
                {.spec = spec, .payload = ConstantValue{}},
                {.spec = spec, .payload = ConstantValue{}},
        };
        const std::vector<GraphValueDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .name = "fused"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, SiluMulParams{}, ConstEvalPolicy{});

        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_EQ(plan->outputs.size(), 1U);
        EXPECT_EQ(plan->outputs[0].spec, spec);
        EXPECT_EQ(plan->outputs[0].nbytes, 2U * static_cast<size_t>(dtype.nbytes()));
    }
}

TEST(ConstEvaluator, SkipsSiluMulUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<DataType> unsupported = {
            DataType::Double(),
            DataType::Int(32),
    };

    for (const DataType dtype: unsupported) {
        const TensorSpec spec = Spec(dtype, {2});
        const std::vector<GraphValueDesc> inputs = {
                {.spec = spec, .payload = ConstantValue{}},
                {.spec = spec, .payload = ConstantValue{}},
        };
        const std::vector<GraphValueDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .name = "fused"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, SiluMulParams{}, ConstEvalPolicy{});

        ASSERT_FALSE(plan.ok());
        EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> gate_bytes =
            BytesFromValues(std::vector<Half>{Half(1.0F), Half(2.0F), Half(3.0F)});
    const std::vector<std::byte> up_bytes =
            BytesFromValues(std::vector<Half>{Half(2.0F), Half(3.0F), Half(4.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Half));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::Float(16), shape, strides),
            TensorView(up_bytes.data(), DataType::Float(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Half> result = ValuesFromBytes<Half>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> gate_f = {1.0F, 2.0F, 3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    for (size_t i = 0; i < 3U; ++i) {
        // Half inputs promote to float for computation, so the expected value
        // follows the same round-trip through float, rounded once on output.
        const float g = static_cast<float>(Half(gate_f[i]));
        const float u = static_cast<float>(Half(up_f[i]));
        float r;
        if (g >= 0.0F) {
            r = g / (1.0F + std::exp(-g));
        } else {
            r = g * std::exp(g) / (1.0F + std::exp(g));
        }
        EXPECT_EQ(result[i], Half(r * u));
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulFloat8E4M3FN) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> gate_bytes = BytesFromValues(
            std::vector<Float8_e4m3fn>{Float8_e4m3fn(1.0F), Float8_e4m3fn(2.0F), Float8_e4m3fn(3.0F)});
    const std::vector<std::byte> up_bytes = BytesFromValues(
            std::vector<Float8_e4m3fn>{Float8_e4m3fn(2.0F), Float8_e4m3fn(3.0F), Float8_e4m3fn(4.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Float8_e4m3fn));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::Float8E4M3FN(), shape, strides),
            TensorView(up_bytes.data(), DataType::Float8E4M3FN(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float8E4M3FN(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Float8_e4m3fn> result = ValuesFromBytes<Float8_e4m3fn>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> gate_f = {1.0F, 2.0F, 3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    for (size_t i = 0; i < 3U; ++i) {
        // fp8 promotes the input to float for computation, so the expected
        // value follows the same round-trip through float.
        const float g = static_cast<float>(Float8_e4m3fn(gate_f[i]));
        const float u = static_cast<float>(Float8_e4m3fn(up_f[i]));
        float r;
        if (g >= 0.0F) {
            r = g / (1.0F + std::exp(-g));
        } else {
            r = g * std::exp(g) / (1.0F + std::exp(g));
        }
        EXPECT_EQ(static_cast<float>(result[i]), static_cast<float>(Float8_e4m3fn(r * u)));
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulFloat8E5M2) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> gate_bytes = BytesFromValues(
            std::vector<Float8_e5m2>{Float8_e5m2(1.0F), Float8_e5m2(2.0F), Float8_e5m2(3.0F)});
    const std::vector<std::byte> up_bytes = BytesFromValues(
            std::vector<Float8_e5m2>{Float8_e5m2(2.0F), Float8_e5m2(3.0F), Float8_e5m2(4.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Float8_e5m2));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::Float8E5M2(), shape, strides),
            TensorView(up_bytes.data(), DataType::Float8E5M2(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float8E5M2(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Float8_e5m2> result = ValuesFromBytes<Float8_e5m2>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> gate_f = {1.0F, 2.0F, 3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    for (size_t i = 0; i < 3U; ++i) {
        const float g = static_cast<float>(Float8_e5m2(gate_f[i]));
        const float u = static_cast<float>(Float8_e5m2(up_f[i]));
        float r;
        if (g >= 0.0F) {
            r = g / (1.0F + std::exp(-g));
        } else {
            r = g * std::exp(g) / (1.0F + std::exp(g));
        }
        EXPECT_EQ(static_cast<float>(result[i]), static_cast<float>(Float8_e5m2(r * u)));
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulBFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> gate_bytes = BytesFromValues(BFloat16Values({0x3F80U, 0x4000U, 0x4040U}));
    const std::vector<std::byte> up_bytes = BytesFromValues(BFloat16Values({0x4000U, 0x4040U, 0x4080U}));
    std::vector<std::byte> output_bytes(3U * sizeof(BFloat16));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::BFloat(16), shape, strides),
            TensorView(up_bytes.data(), DataType::BFloat(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::BFloat(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<BFloat16> result = ValuesFromBytes<BFloat16>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    // bf16 values 1.0, 2.0, 3.0 (0x3F80, 0x4000, 0x4040) and 2.0, 3.0, 4.0
    // (0x4000, 0x4040, 0x4080) are exact; expected follows the float
    // round-trip then rounds once on output, compared by raw bits.
    const std::vector<float> gate_f = {1.0F, 2.0F, 3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    std::vector<BFloat16> expected;
    expected.reserve(3);
    for (size_t i = 0; i < 3U; ++i) {
        const float g = static_cast<float>(BFloat16(gate_f[i]));
        const float u = static_cast<float>(BFloat16(up_f[i]));
        float r;
        if (g >= 0.0F) {
            r = g / (1.0F + std::exp(-g));
        } else {
            r = g * std::exp(g) / (1.0F + std::exp(g));
        }
        expected.emplace_back(r * u);
    }
    EXPECT_EQ(BFloat16Bits(result), BFloat16Bits(expected));
}

TEST(ConstEvaluator, EvaluatesSiluMulNonContiguousInput) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{2};
    const std::vector<int64_t> noncontig_strides{2};
    const std::vector<int64_t> output_strides{1};
    const std::vector<std::byte> gate_bytes = BytesFromValues<float>({1.0F, 10.0F, 2.0F, 20.0F});
    const std::vector<std::byte> up_bytes = BytesFromValues<float>({3.0F, 30.0F, 4.0F, 40.0F});
    std::vector<std::byte> output_bytes(2U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::Float32(), shape, noncontig_strides),
            TensorView(up_bytes.data(), DataType::Float32(), shape, noncontig_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 2U);
    const std::vector<float> gate_f = {1.0F, 2.0F};
    const std::vector<float> up_f = {3.0F, 4.0F};
    for (size_t i = 0; i < 2U; ++i) {
        const float g = gate_f[i];
        float r;
        if (g >= 0.0F) {
            r = g / (1.0F + std::exp(-g));
        } else {
            r = g * std::exp(g) / (1.0F + std::exp(g));
        }
        EXPECT_NEAR(result[i], r * up_f[i], 1e-5F);
    }
}

TEST(ConstEvaluator, EvaluatesSiluMulNegativeGate) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> gate_bytes = BytesFromValues<float>({-1.0F, -2.0F, -3.0F});
    const std::vector<std::byte> up_bytes = BytesFromValues<float>({2.0F, 3.0F, 4.0F});
    std::vector<std::byte> output_bytes(3U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(gate_bytes.data(), DataType::Float32(), shape, strides),
            TensorView(up_bytes.data(), DataType::Float32(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> gate_f = {-1.0F, -2.0F, -3.0F};
    const std::vector<float> up_f = {2.0F, 3.0F, 4.0F};
    for (size_t i = 0; i < 3U; ++i) {
        const float g = gate_f[i];
        // Negative gate exercises the overflow-safe exp(x) branch.
        const float exp_g = std::exp(g);
        const float r = g * exp_g / (1.0F + exp_g);
        EXPECT_NEAR(result[i], r * up_f[i], 1e-5F);
    }
}

}// namespace