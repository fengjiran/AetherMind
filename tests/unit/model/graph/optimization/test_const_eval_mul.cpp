#include "../test_graph_helpers.h"
#include "test_const_eval_helpers.h"

#include <gtest/gtest.h>

#include <limits>
#include <type_traits>

namespace aethermind {
namespace {

using namespace test_utils;

// ── ElementwiseMul evaluator tests ──

template<typename T>
void ExpectMulEvaluation(DataType dtype,
                         std::vector<T> lhs,
                         std::vector<T> rhs,
                         std::vector<T> expected) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{static_cast<int64_t>(lhs.size())};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues(std::move(lhs));
    const std::vector<std::byte> rhs_bytes = BytesFromValues(std::move(rhs));
    std::vector<std::byte> output_bytes(expected.size() * sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), dtype, shape, strides),
            TensorView(rhs_bytes.data(), dtype, shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_DOUBLE_EQ(static_cast<double>(result[i]), static_cast<double>(expected[i]));
        } else {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
}

template<typename T>
void ExpectMulStridedEvaluation(DataType dtype,
                                std::vector<T> lhs,
                                std::vector<int64_t> shape,
                                std::vector<int64_t> lhs_strides,
                                std::vector<T> rhs,
                                std::vector<int64_t> rhs_strides,
                                std::vector<T> expected) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> output_strides = MakeContiguousStridesOrDie(shape);
    const std::vector<std::byte> lhs_bytes = BytesFromValues(std::move(lhs));
    const std::vector<std::byte> rhs_bytes = BytesFromValues(std::move(rhs));
    std::vector<std::byte> output_bytes(expected.size() * sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), dtype, shape, lhs_strides),
            TensorView(rhs_bytes.data(), dtype, shape, rhs_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        if constexpr (std::is_floating_point_v<T>) {
            EXPECT_DOUBLE_EQ(static_cast<double>(result[i]), static_cast<double>(expected[i]));
        } else {
            EXPECT_EQ(result[i], expected[i]);
        }
    }
}

TEST(ConstEvaluator, FindsMulEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kElementwiseMul), nullptr);
}

TEST(ConstEvaluator, PlansMulFloat32SameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {2});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(float));
    EXPECT_EQ(plan->outputs[0].debug_name, "folded_product");
}

TEST(ConstEvaluator, PlansMulSupportedDTypesSameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<DataType> dtypes = {
            DataType::Float32(),
            DataType::Double(),
            DataType::Int(32),
            DataType::Int(64),
    };

    for (const DataType dtype: dtypes) {
        const TensorSpec spec = Spec(dtype, {2});
        const std::vector<GraphValueDesc> inputs = {
                {.spec = spec, .payload = ConstantValue{}},
                {.spec = spec, .payload = ConstantValue{}},
        };
        const std::vector<GraphValueDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .name = "product"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_EQ(plan->outputs.size(), 1U);
        EXPECT_EQ(plan->outputs[0].spec, spec);
        EXPECT_EQ(plan->outputs[0].nbytes, 2U * static_cast<size_t>(dtype.nbytes()));
    }
}

TEST(ConstEvaluator, PlansMulBFloat16SameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::BFloat(16), {2});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(BFloat16));
}

TEST(ConstEvaluator, SkipsMulMismatchedShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {3}), .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, SkipsMulRankZeroScalarTensorMismatch) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, SkipsMulUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float(16), {2});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, PlansMulComputeBudgetExceeded) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    constexpr int64_t kExceed = int64_t{64U} * 1024 + 1;
    const TensorSpec spec = Spec(DataType::Float32(), {kExceed});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, PlansMulOutputByteBudgetExceeded) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    // Float64 = 8 bytes, 8193 elements = 65544 bytes > 64K byte budget
    const std::vector<int64_t> shape{8193};
    const TensorSpec spec = Spec(DataType::Double(), shape);
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, EvaluatesMulFloat32) {
    ExpectMulEvaluation<float>(DataType::Float32(), {2.0F, 3.0F}, {4.0F, 5.0F}, {8.0F, 15.0F});
}

TEST(ConstEvaluator, EvaluatesMulFloat64) {
    ExpectMulEvaluation<double>(DataType::Double(), {2.25, 3.5}, {4.0, 2.0}, {9.0, 7.0});
}

TEST(ConstEvaluator, EvaluatesMulInt32) {
    ExpectMulEvaluation<int32_t>(DataType::Int(32), {-3, 2, 7}, {5, -4, 8}, {-15, -8, 56});
}

TEST(ConstEvaluator, EvaluatesMulInt64) {
    ExpectMulEvaluation<int64_t>(DataType::Int(64), {-3, 2, 7}, {5, -4, 8}, {-15, -8, 56});
}

TEST(ConstEvaluator, EvaluatesMulBFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{2};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues(BFloat16Values({0x4000U, 0x4040U}));
    const std::vector<std::byte> rhs_bytes = BytesFromValues(BFloat16Values({0x4000U, 0x4000U}));
    std::vector<std::byte> output_bytes(2U * sizeof(BFloat16));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::BFloat(16), shape, strides),
            TensorView(rhs_bytes.data(), DataType::BFloat(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::BFloat(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(BFloat16Bits(ValuesFromBytes<BFloat16>(output_bytes)),
              (std::vector<uint16_t>{0x4080U, 0x40C0U}));
}

TEST(ConstEvaluator, EvaluatesMulNonContiguousSameShape) {
    // 2x2 matrix with row stride 3, simulating a 2x2 slice of 2x3 storage.
    // lhs elements at offsets {0,1,3,4} = {1,10,2,30}
    // rhs elements at offsets {0,1,3,4} = {3,30,4,50}
    const std::vector<int64_t> shape{2, 2};
    const std::vector<int64_t> noncontig_strides{3, 1};
    ExpectMulStridedEvaluation<float>(DataType::Float32(),
                                      {1.0F, 10.0F, 20.0F, 2.0F, 30.0F, 40.0F},
                                      shape,
                                      noncontig_strides,
                                      {3.0F, 30.0F, 40.0F, 4.0F, 50.0F, 60.0F},
                                      noncontig_strides,
                                      {3.0F, 300.0F, 8.0F, 1500.0F});
}

TEST(ConstEvaluator, SkipsMulInt32Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{1};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int32_t>({std::numeric_limits<int32_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int32_t>({2});
    std::vector<std::byte> output_bytes(sizeof(int32_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(32), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Int(32), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(32), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

TEST(ConstEvaluator, SkipsMulInt64Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{1};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int64_t>({std::numeric_limits<int64_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int64_t>({2});
    std::vector<std::byte> output_bytes(sizeof(int64_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(64), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Int(64), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(64), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

// ── ElementwiseMul rank-zero scalar tests ──

TEST(ConstEvaluator, PlansMulRankZeroScalarPlusScalar) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .name = "product"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, static_cast<size_t>(DataType::Float32().nbytes()));
    EXPECT_TRUE(plan->outputs[0].strides.empty());
}

template<typename T>
void ExpectMulRankZeroEvaluation(DataType dtype, T lhs_value, T rhs_value, T expected_value) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{};
    const std::vector<int64_t> strides{};
    const std::vector<std::byte> lhs_bytes = BytesFromValues(std::vector<T>{lhs_value});
    const std::vector<std::byte> rhs_bytes = BytesFromValues(std::vector<T>{rhs_value});
    std::vector<std::byte> output_bytes(sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), dtype, shape, strides),
            TensorView(rhs_bytes.data(), dtype, shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, ElementwiseMulParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), 1U);
    if constexpr (std::is_floating_point_v<T>) {
        EXPECT_DOUBLE_EQ(static_cast<double>(result[0]), static_cast<double>(expected_value));
    } else {
        EXPECT_EQ(result[0], expected_value);
    }
}

TEST(ConstEvaluator, EvaluatesMulRankZeroScalarFloat32) {
    ExpectMulRankZeroEvaluation<float>(DataType::Float32(), 4.0F, 3.0F, 12.0F);
}

TEST(ConstEvaluator, EvaluatesMulRankZeroScalarBFloat16) {
    ExpectMulRankZeroEvaluation<BFloat16>(
            DataType::BFloat(16),
            BFloat16(2.0F),
            BFloat16(3.0F),
            BFloat16(6.0F));
}

TEST(ConstEvaluator, EvaluatesMulRankZeroScalarInt32) {
    ExpectMulRankZeroEvaluation<int32_t>(DataType::Int(32), -3, 5, -15);
}

// Mul rank-zero adversarial — unsupported dtype
TEST(ConstEvaluator, SkipsMulRankZeroUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kElementwiseMul);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float(16), {});
    const std::vector<GraphValueDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<GraphValueDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, ElementwiseMulParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
