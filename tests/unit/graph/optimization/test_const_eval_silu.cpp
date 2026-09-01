#include "../test_graph_helpers.h"
#include "test_const_eval_helpers.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;
using namespace test_utils;

TEST(ConstEvaluator, FindsSiluEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kSilu), nullptr);
}

template<typename T>
void ExpectSiluEvaluation(DataType dtype,
                          std::vector<T> input,
                          std::vector<T> expected) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{static_cast<int64_t>(input.size())};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> input_bytes = BytesFromValues(std::move(input));
    std::vector<std::byte> output_bytes(expected.size() * sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), dtype, shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_same_v<T, float>) {
            EXPECT_NEAR(result[i], expected[i], 1e-5F);
        } else {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
}

TEST(ConstEvaluator, PlansSiluFloat32SameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {3});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "act"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, 3U * sizeof(float));
    EXPECT_EQ(plan->outputs[0].name, "folded_act");
}

TEST(ConstEvaluator, PlansSiluSupportedDTypesSameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
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
        };
        const std::vector<GraphValueDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .name = "act"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_EQ(plan->outputs.size(), 1U);
        EXPECT_EQ(plan->outputs[0].spec, spec);
        EXPECT_EQ(plan->outputs[0].nbytes, 2U * static_cast<size_t>(dtype.nbytes()));
    }
}

TEST(ConstEvaluator, SkipsSiluUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<DataType> unsupported = {
            DataType::Double(),
            DataType::Int(32),
    };

    for (const DataType dtype: unsupported) {
        const TensorSpec spec = Spec(dtype, {2});
        const std::vector<GraphValueDesc> inputs = {
                {.spec = spec, .payload = ConstantValue{}},
        };
        const std::vector<GraphValueDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .name = "act"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

        ASSERT_FALSE(plan.ok());
        EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
    }
}

TEST(ConstEvaluator, SkipsSiluMismatchedShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {3}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, SkipsSiluRankZeroInputOutputMismatch) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, PlansSiluCostOverComputeBudget) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    // bf16: 32768 elements × 2 bytes = 65536 == max_output_bytes (byte gate
    // passes, strict >), compute = 3 × 32768 = 98304 > 64K (compute gate
    // fails) — isolates the compute budget independently of the byte budget.
    const std::vector<int64_t> shape{32768};
    const TensorSpec spec = Spec(DataType::BFloat(16), shape);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "act"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->cost.compute_ops, 3U * 32768U);
    EXPECT_EQ(plan->cost.output_bytes, 65536U);
    EXPECT_EQ(CheckFoldingBudget(plan->cost, ConstEvalPolicy{}).code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, PlansSiluCostOverOutputByteBudget) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    // f32: 16385 elements × 4 bytes = 65540 > 65536 (byte gate fails);
    // compute = 3 × 16385 = 49155 < 65536 (compute gate passes) — isolates
    // the byte budget independently of the compute budget.
    const std::vector<int64_t> shape{16385};
    const TensorSpec spec = Spec(DataType::Float32(), shape);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "act"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->cost.compute_ops, 3U * 16385U);
    EXPECT_EQ(plan->cost.output_bytes, 65540U);
    EXPECT_EQ(CheckFoldingBudget(plan->cost, ConstEvalPolicy{}).code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, EvaluatesSiluFloat32) {
    ExpectSiluEvaluation<float>(DataType::Float32(),
                                {0.0F, 1.0F, 2.0F, -1.0F, -2.0F},
                                {0.0F,
                                 0.7310586F,
                                 1.7615942F,
                                 -0.26894143F,
                                 -0.23840584F});
}

TEST(ConstEvaluator, EvaluatesSiluFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> input_bytes =
            BytesFromValues(std::vector<Half>{Half(1.0F), Half(2.0F), Half(3.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Half));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::Float(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Half> result = ValuesFromBytes<Half>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    std::vector<Half> expected;
    for (float x: {1.0F, 2.0F, 3.0F}) {
        float r;
        if (x >= 0.0F) {
            r = x / (1.0F + std::exp(-x));
        } else {
            r = x * std::exp(x) / (1.0F + std::exp(x));
        }
        expected.emplace_back(r);
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(result[i], expected[i]);
    }
}

TEST(ConstEvaluator, EvaluatesSiluFloat8E4M3FN) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> input_bytes =
            BytesFromValues(std::vector<Float8_e4m3fn>{Float8_e4m3fn(1.0F), Float8_e4m3fn(2.0F), Float8_e4m3fn(3.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Float8_e4m3fn));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::Float8E4M3FN(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float8E4M3FN(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Float8_e4m3fn> result = ValuesFromBytes<Float8_e4m3fn>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> input_f = {1.0F, 2.0F, 3.0F};
    for (size_t i = 0; i < 3U; ++i) {
        // fp8 promotes the input to float for computation, so the expected
        // value follows the same round-trip through float.
        const float x = static_cast<float>(Float8_e4m3fn(input_f[i]));
        float r;
        if (x >= 0.0F) {
            r = x / (1.0F + std::exp(-x));
        } else {
            r = x * std::exp(x) / (1.0F + std::exp(x));
        }
        EXPECT_EQ(static_cast<float>(result[i]), static_cast<float>(Float8_e4m3fn(r)));
    }
}

TEST(ConstEvaluator, EvaluatesSiluFloat8E5M2) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> input_bytes =
            BytesFromValues(std::vector<Float8_e5m2>{Float8_e5m2(1.0F), Float8_e5m2(2.0F), Float8_e5m2(3.0F)});
    std::vector<std::byte> output_bytes(3U * sizeof(Float8_e5m2));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::Float8E5M2(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float8E5M2(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<Float8_e5m2> result = ValuesFromBytes<Float8_e5m2>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    const std::vector<float> input_f = {1.0F, 2.0F, 3.0F};
    for (size_t i = 0; i < 3U; ++i) {
        const float x = static_cast<float>(Float8_e5m2(input_f[i]));
        float r;
        if (x >= 0.0F) {
            r = x / (1.0F + std::exp(-x));
        } else {
            r = x * std::exp(x) / (1.0F + std::exp(x));
        }
        EXPECT_EQ(static_cast<float>(result[i]), static_cast<float>(Float8_e5m2(r)));
    }
}

TEST(ConstEvaluator, EvaluatesSiluBFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> input_bytes = BytesFromValues(BFloat16Values({0x3F80U, 0x4000U, 0x4040U}));
    std::vector<std::byte> output_bytes(3U * sizeof(BFloat16));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::BFloat(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::BFloat(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<BFloat16> result = ValuesFromBytes<BFloat16>(output_bytes);
    const std::vector<float> input_f = {1.0F, 2.0F, 3.0F};
    std::vector<BFloat16> expected;
    expected.reserve(3);
    for (float x: input_f) {
        float r;
        if (x >= 0.0F) {
            r = x / (1.0F + std::exp(-x));
        } else {
            r = x * std::exp(x) / (1.0F + std::exp(x));
        }
        expected.emplace_back(r);
    }
    EXPECT_EQ(BFloat16Bits(result), BFloat16Bits(expected));
}

TEST(ConstEvaluator, EvaluatesSiluNonContiguousInput) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{2};
    const std::vector<int64_t> noncontig_strides{2};
    const std::vector<int64_t> output_strides{1};
    const std::vector<std::byte> input_bytes = BytesFromValues<float>({1.0F, 10.0F, 2.0F, 20.0F});
    std::vector<std::byte> output_bytes(2U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::Float32(), shape, noncontig_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 2U);
    EXPECT_NEAR(result[0], 0.7310586F, 1e-5F);
    EXPECT_NEAR(result[1], 1.7615942F, 1e-5F);
}

TEST(ConstEvaluator, EvaluatesSiluSpecialValues) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{5};
    const std::vector<int64_t> strides{1};
    const auto pos_inf = std::numeric_limits<float>::infinity();
    const auto neg_inf = -pos_inf;
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<std::byte> input_bytes = BytesFromValues<float>({pos_inf, neg_inf, nan, 0.0F, -0.0F});
    std::vector<std::byte> output_bytes(5U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), DataType::Float32(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 5U);
    EXPECT_TRUE(std::isinf(result[0]) && result[0] > 0.0F);
    EXPECT_TRUE(std::isnan(result[1]));
    EXPECT_TRUE(std::isnan(result[2]));
    EXPECT_EQ(result[3], 0.0F);
    EXPECT_FALSE(std::signbit(result[3]));
    EXPECT_EQ(result[4], -0.0F);
    EXPECT_TRUE(std::signbit(result[4]));
}

// ── Silu rank-zero scalar tests ──

TEST(ConstEvaluator, PlansSiluRankZero) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "act"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, SiluParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, static_cast<size_t>(DataType::Float32().nbytes()));
    EXPECT_TRUE(plan->outputs[0].strides.empty());
}

template<typename T>
void ExpectSiluRankZeroEvaluation(DataType dtype, T input_value, T expected_value) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{};
    const std::vector<int64_t> strides{};
    const std::vector<std::byte> input_bytes = BytesFromValues(std::vector<T>{input_value});
    std::vector<std::byte> output_bytes(sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(input_bytes.data(), dtype, shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), 1U);
    if constexpr (std::is_same_v<T, float>) {
        EXPECT_NEAR(result[0], expected_value, 1e-5F);
    } else {
        EXPECT_EQ(result[0], expected_value);
    }
}

TEST(ConstEvaluator, EvaluatesSiluRankZeroFloat32) {
    float x = 2.0F;
    float expected;
    if (x >= 0.0F) {
        expected = x / (1.0F + std::exp(-x));
    } else {
        expected = x * std::exp(x) / (1.0F + std::exp(x));
    }
    ExpectSiluRankZeroEvaluation<float>(DataType::Float32(), x, expected);
}

TEST(ConstEvaluator, EvaluatesSiluRankZeroBFloat16) {
    float x = 2.0F;
    float expected;
    if (x >= 0.0F) {
        expected = x / (1.0F + std::exp(-x));
    } else {
        expected = x * std::exp(x) / (1.0F + std::exp(x));
    }
    ExpectSiluRankZeroEvaluation<BFloat16>(
            DataType::BFloat(16),
            BFloat16(2.0F),
            BFloat16(expected));
}

} // namespace
