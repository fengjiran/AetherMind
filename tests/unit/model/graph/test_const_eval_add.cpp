#include "test_const_eval_helpers.h"
#include "test_graph_helpers.h"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>

namespace aethermind {
namespace {

using namespace test_utils;

template<typename T>
void ExpectAddEvaluation(DataType dtype,
                         std::vector<T> lhs,
                         std::vector<T> rhs,
                         std::vector<T> expected) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
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

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

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
void ExpectAddBroadcastEvaluation(DataType dtype,
                                  std::vector<T> lhs,
                                  std::vector<int64_t> lhs_shape,
                                  std::vector<T> rhs,
                                  std::vector<int64_t> rhs_shape,
                                  std::vector<T> expected,
                                  std::vector<int64_t> output_shape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> lhs_strides = MakeContiguousStridesOrDie(lhs_shape);
    const std::vector<int64_t> rhs_strides = MakeContiguousStridesOrDie(rhs_shape);
    const std::vector<int64_t> output_strides = MakeContiguousStridesOrDie(output_shape);
    const std::vector<std::byte> lhs_bytes = BytesFromValues(std::move(lhs));
    const std::vector<std::byte> rhs_bytes = BytesFromValues(std::move(rhs));
    std::vector<std::byte> output_bytes(expected.size() * sizeof(T));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), dtype, lhs_shape, lhs_strides),
            TensorView(rhs_bytes.data(), dtype, rhs_shape, rhs_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), dtype, output_shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

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

TEST(ConstEvaluator, FindsAddEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kAdd), nullptr);
}

TEST(ConstEvaluator, PlansAddFloat32SameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {2});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(float));
    EXPECT_EQ(plan->outputs[0].debug_name, "folded_sum");
}

TEST(ConstEvaluator, PlansAddSupportedDTypesSameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<DataType> dtypes = {
            DataType::Float32(),
            DataType::Double(),
            DataType::Int(32),
            DataType::Int(64),
    };

    for (const DataType dtype: dtypes) {
        const TensorSpec spec = Spec(dtype, {2});
        const std::vector<NodeOutputDesc> inputs = {
                {.spec = spec, .payload = ConstantValue{}},
                {.spec = spec, .payload = ConstantValue{}},
        };
        const std::vector<NodeOutputDesc> outputs = {
                {.spec = spec, .payload = ActivationValue{}, .debug_name = "sum"},
        };

        const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

        ASSERT_TRUE(plan.ok()) << plan.status().ToString();
        ASSERT_EQ(plan->outputs.size(), 1U);
        EXPECT_EQ(plan->outputs[0].spec, spec);
        EXPECT_EQ(plan->outputs[0].nbytes, 2U * static_cast<size_t>(dtype.nbytes()));
    }
}

TEST(ConstEvaluator, PlansAddBFloat16SameShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::BFloat(16), {2});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(BFloat16));
}

TEST(ConstEvaluator, PlansAddBroadcastRowVector) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec output = Spec(DataType::Float32(), {2, 3});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = output, .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {3}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = output, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, output);
    EXPECT_EQ(plan->outputs[0].nbytes, 6U * sizeof(float));
}

TEST(ConstEvaluator, PlansAddBroadcastBothInputs) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec output = Spec(DataType::Float32(), {2, 3});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2, 1}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {1, 3}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = output, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, output);
}

TEST(ConstEvaluator, SkipsAddBroadcastOutputMismatch) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2, 1}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {1, 3}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2, 1}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, PlansAddScalarBroadcastRankZeroLhs) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec output = Spec(DataType::Float32(), {2});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = output, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, output);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(float));
}

TEST(ConstEvaluator, PlansAddScalarBroadcastRankZeroRhs) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec output = Spec(DataType::Float32(), {2});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = output, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, output);
    EXPECT_EQ(plan->outputs[0].nbytes, 2U * sizeof(float));
}

TEST(ConstEvaluator, SkipsAddUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float(16), {2});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, SkipsAddMismatchedShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {3}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

TEST(ConstEvaluator, EvaluatesAddFloat32) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{3};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<float>({1.0F, 2.0F, 3.0F});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<float>({4.0F, 5.0F, 6.0F});
    std::vector<std::byte> output_bytes(3U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Float32(), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Float32(), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    std::vector<float> result(3);
    std::memcpy(result.data(), output_bytes.data(), output_bytes.size());
    EXPECT_FLOAT_EQ(result[0], 5.0F);
    EXPECT_FLOAT_EQ(result[1], 7.0F);
    EXPECT_FLOAT_EQ(result[2], 9.0F);
}

TEST(ConstEvaluator, EvaluatesAddFloat64) {
    ExpectAddEvaluation<double>(DataType::Double(), {1.25, 2.5}, {3.5, 4.75}, {4.75, 7.25});
}

TEST(ConstEvaluator, EvaluatesAddInt32) {
    ExpectAddEvaluation<int32_t>(DataType::Int(32), {-3, 2, 7}, {5, -4, 8}, {2, -2, 15});
}

TEST(ConstEvaluator, EvaluatesAddInt64) {
    ExpectAddEvaluation<int64_t>(DataType::Int(64), {-3, 2, 7}, {5, -4, 8}, {2, -2, 15});
}

TEST(ConstEvaluator, EvaluatesAddBFloat16) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{2};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues(BFloat16Values({0x3F80U, 0x4000U}));
    const std::vector<std::byte> rhs_bytes = BytesFromValues(BFloat16Values({0x4040U, 0x4080U}));
    std::vector<std::byte> output_bytes(2U * sizeof(BFloat16));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::BFloat(16), shape, strides),
            TensorView(rhs_bytes.data(), DataType::BFloat(16), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::BFloat(16), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(BFloat16Bits(ValuesFromBytes<BFloat16>(output_bytes)),
              (std::vector<uint16_t>{0x4080U, 0x40C0U}));
}

TEST(ConstEvaluator, EvaluatesAddBroadcastRowVector) {
    ExpectAddBroadcastEvaluation<float>(DataType::Float32(),
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                        {2, 3},
                                        {10.0F, 20.0F, 30.0F},
                                        {3},
                                        {11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F},
                                        {2, 3});
}

TEST(ConstEvaluator, EvaluatesAddBroadcastBothInputs) {
    ExpectAddBroadcastEvaluation<float>(DataType::Float32(),
                                        {1.0F, 2.0F},
                                        {2, 1},
                                        {10.0F, 20.0F, 30.0F},
                                        {1, 3},
                                        {11.0F, 21.0F, 31.0F, 12.0F, 22.0F, 32.0F},
                                        {2, 3});
}

TEST(ConstEvaluator, EvaluatesAddBroadcastSingleElement) {
    ExpectAddBroadcastEvaluation<float>(DataType::Float32(),
                                        {5.0F},
                                        {1},
                                        {1.0F, 2.0F, 3.0F, 4.0F},
                                        {2, 2},
                                        {6.0F, 7.0F, 8.0F, 9.0F},
                                        {2, 2});
}

TEST(ConstEvaluator, EvaluatesAddBroadcastReversedInputs) {
    ExpectAddBroadcastEvaluation<float>(DataType::Float32(),
                                        {10.0F, 20.0F, 30.0F},
                                        {3},
                                        {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F},
                                        {2, 3},
                                        {11.0F, 22.0F, 33.0F, 14.0F, 25.0F, 36.0F},
                                        {2, 3});
}

TEST(ConstEvaluator, SkipsAddInt32Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{1};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int32_t>({std::numeric_limits<int32_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int32_t>({1});
    std::vector<std::byte> output_bytes(sizeof(int32_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(32), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Int(32), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(32), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

TEST(ConstEvaluator, SkipsAddBroadcastInt32Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> lhs_shape{1};
    const std::vector<int64_t> rhs_shape{2};
    const std::vector<int64_t> output_shape{2};
    const std::vector<int64_t> lhs_strides = MakeContiguousStridesOrDie(lhs_shape);
    const std::vector<int64_t> rhs_strides = MakeContiguousStridesOrDie(rhs_shape);
    const std::vector<int64_t> output_strides = MakeContiguousStridesOrDie(output_shape);
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int32_t>({std::numeric_limits<int32_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int32_t>({0, 1});
    std::vector<std::byte> output_bytes(2U * sizeof(int32_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(32), lhs_shape, lhs_strides),
            TensorView(rhs_bytes.data(), DataType::Int(32), rhs_shape, rhs_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(32), output_shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

TEST(ConstEvaluator, SkipsAddInt64Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{1};
    const std::vector<int64_t> strides{1};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int64_t>({std::numeric_limits<int64_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int64_t>({1});
    std::vector<std::byte> output_bytes(sizeof(int64_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(64), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Int(64), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(64), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

// ── Add rank-zero scalar tests ──

TEST(ConstEvaluator, PlansAddRankZeroScalarPlusScalar) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float32(), {});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}, .debug_name = "sum"},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->outputs.size(), 1U);
    EXPECT_EQ(plan->outputs[0].spec, spec);
    EXPECT_EQ(plan->outputs[0].nbytes, static_cast<size_t>(DataType::Float32().nbytes()));
    EXPECT_TRUE(plan->outputs[0].strides.empty());
}

template<typename T>
void ExpectAddRankZeroEvaluation(DataType dtype, T lhs_value, T rhs_value, T expected_value) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
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

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<T> result = ValuesFromBytes<T>(output_bytes);
    ASSERT_EQ(result.size(), 1U);
    if constexpr (std::is_floating_point_v<T>) {
        EXPECT_DOUBLE_EQ(static_cast<double>(result[0]), static_cast<double>(expected_value));
    } else {
        EXPECT_EQ(result[0], expected_value);
    }
}

TEST(ConstEvaluator, EvaluatesAddRankZeroScalarFloat32) {
    ExpectAddRankZeroEvaluation<float>(DataType::Float32(), 3.5F, 2.25F, 5.75F);
}

TEST(ConstEvaluator, EvaluatesAddRankZeroScalarBFloat16) {
    ExpectAddRankZeroEvaluation<BFloat16>(
            DataType::BFloat(16),
            BFloat16(1.5F),
            BFloat16(2.5F),
            BFloat16(4.0F));
}

TEST(ConstEvaluator, EvaluatesAddRankZeroScalarInt32) {
    ExpectAddRankZeroEvaluation<int32_t>(DataType::Int(32), -7, 15, 8);
}

// Add scalar broadcast Evaluate tests — rank-zero lhs/tensor rhs
TEST(ConstEvaluator, EvaluatesAddScalarBroadcastRankZeroLhs) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> lhs_shape{};
    const std::vector<int64_t> lhs_strides{};
    const std::vector<int64_t> rhs_shape{3};
    const std::vector<int64_t> rhs_strides{1};
    const std::vector<int64_t> output_shape{3};
    const std::vector<int64_t> output_strides = MakeContiguousStridesOrDie(output_shape);
    const std::vector<std::byte> lhs_bytes = BytesFromValues<float>({2.0F});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<float>({10.0F, 20.0F, 30.0F});
    std::vector<std::byte> output_bytes(3U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Float32(), lhs_shape, lhs_strides),
            TensorView(rhs_bytes.data(), DataType::Float32(), rhs_shape, rhs_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), output_shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    EXPECT_FLOAT_EQ(result[0], 12.0F);
    EXPECT_FLOAT_EQ(result[1], 22.0F);
    EXPECT_FLOAT_EQ(result[2], 32.0F);
}

// Add scalar broadcast Evaluate — rank-zero rhs/tensor lhs
TEST(ConstEvaluator, EvaluatesAddScalarBroadcastRankZeroRhs) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> lhs_shape{3};
    const std::vector<int64_t> lhs_strides{1};
    const std::vector<int64_t> rhs_shape{};
    const std::vector<int64_t> rhs_strides{};
    const std::vector<int64_t> output_shape{3};
    const std::vector<int64_t> output_strides = MakeContiguousStridesOrDie(output_shape);
    const std::vector<std::byte> lhs_bytes = BytesFromValues<float>({10.0F, 20.0F, 30.0F});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<float>({2.0F});
    std::vector<std::byte> output_bytes(3U * sizeof(float));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Float32(), lhs_shape, lhs_strides),
            TensorView(rhs_bytes.data(), DataType::Float32(), rhs_shape, rhs_strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Float32(), output_shape, output_strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    const std::vector<float> result = ValuesFromBytes<float>(output_bytes);
    ASSERT_EQ(result.size(), 3U);
    EXPECT_FLOAT_EQ(result[0], 12.0F);
    EXPECT_FLOAT_EQ(result[1], 22.0F);
    EXPECT_FLOAT_EQ(result[2], 32.0F);
}

// Add rank-zero adversarial — unsupported dtype
TEST(ConstEvaluator, SkipsAddRankZeroUnsupportedDType) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const TensorSpec spec = Spec(DataType::Float(16), {});
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = spec, .payload = ConstantValue{}},
            {.spec = spec, .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = spec, .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
}

// Add rank-zero adversarial — integer overflow
TEST(ConstEvaluator, SkipsAddRankZeroInt32Overflow) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<int64_t> shape{};
    const std::vector<int64_t> strides{};
    const std::vector<std::byte> lhs_bytes = BytesFromValues<int32_t>({std::numeric_limits<int32_t>::max()});
    const std::vector<std::byte> rhs_bytes = BytesFromValues<int32_t>({1});
    std::vector<std::byte> output_bytes(sizeof(int32_t));
    const std::vector<TensorView> inputs = {
            TensorView(lhs_bytes.data(), DataType::Int(32), shape, strides),
            TensorView(rhs_bytes.data(), DataType::Int(32), shape, strides),
    };
    std::vector<MutableTensorView> outputs = {
            MutableTensorView(output_bytes.data(), DataType::Int(32), shape, strides),
    };

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    EXPECT_EQ(status.code(), StatusCode::kOverflow);
}

}// namespace
}// namespace aethermind
