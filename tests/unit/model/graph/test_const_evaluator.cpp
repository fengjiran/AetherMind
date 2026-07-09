#include "aethermind/model/graph/const_evaluator.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace aethermind {
namespace {

TensorSpec Spec(DataType dtype, std::vector<int64_t> shape) {
    return {.dtype = dtype, .shape = SymbolicShape(IntArrayView(shape))};
}

template<typename T>
std::vector<std::byte> BytesFromValues(std::vector<T> values) {
    std::vector<std::byte> bytes(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    return bytes;
}

template<typename T>
std::vector<T> ValuesFromBytes(const std::vector<std::byte>& bytes) {
    std::vector<T> values(bytes.size() / sizeof(T));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

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

TEST(ConstEvaluator, FindsAddEvaluator) {
    EXPECT_NE(FindConstEvaluator(OpType::kAdd), nullptr);
    EXPECT_EQ(FindConstEvaluator(OpType::kSilu), nullptr);
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

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
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

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
}

}// namespace
}// namespace aethermind
