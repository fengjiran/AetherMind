#include "test_const_eval_helpers.h"
#include "test_graph_helpers.h"

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

namespace aethermind {
namespace {

using namespace test_utils;

TEST(ConstEvaluator, FindsSiluMulEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kSiluMul), nullptr);
}

// ── SiluMul rank-zero scalar tests ──

TEST(ConstEvaluator, PlansSiluMulRankZero) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSiluMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .debug_name = "fused"},
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
    } else if constexpr (std::is_same_v<T, double>) {
        EXPECT_NEAR(result[0], expected_value, 1e-12);
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
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
