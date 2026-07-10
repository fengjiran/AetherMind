#include "aethermind/model/graph/const_evaluator.h"

#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include <limits>
#include <type_traits>
#include <vector>

namespace aethermind {
namespace {

// Unwraps MakeContiguousStrides or fatally aborts the test.
// Test shapes are crafted to avoid overflow; a failure here indicates
// a bug in the test fixture, not the code under test.
std::vector<int64_t> MakeContiguousStridesOrDie(std::span<const int64_t> shape) {
    auto result = MakeContiguousStrides(shape);
    AM_CHECK(result.ok(), "Test helper: MakeContiguousStrides failed: {}", result.status().ToString());
    return std::move(*result);
}

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

std::vector<uint16_t> BFloat16Bits(const std::vector<BFloat16>& values) {
    std::vector<uint16_t> bits;
    bits.reserve(values.size());
    for (const BFloat16 value: values) {
        bits.push_back(value.x);
    }
    return bits;
}

std::vector<BFloat16> BFloat16Values(std::vector<uint16_t> bits) {
    std::vector<BFloat16> values;
    values.reserve(bits.size());
    for (const uint16_t value: bits) {
        values.emplace_back(value, BFloat16::from_bits());
    }
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

TEST(ConstEvaluator, SkipsAddRankZeroScalarShape) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);
    const std::vector<NodeOutputDesc> inputs = {
            {.spec = Spec(DataType::Float32(), {}), .payload = ConstantValue{}},
            {.spec = Spec(DataType::Float32(), {2}), .payload = ConstantValue{}},
    };
    const std::vector<NodeOutputDesc> outputs = {
            {.spec = Spec(DataType::Float32(), {2}), .payload = ActivationValue{}},
    };

    const auto plan = evaluator->Plan(inputs, outputs, AddParams{}, ConstEvalPolicy{});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kUnimplemented);
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

    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
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

// ── MakeContiguousStrides helper tests ──

TEST(ConstEvaluator, MakeContiguousStridesEmpty) {
    const auto result = MakeContiguousStrides(std::span<const int64_t>{});
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_TRUE(result->empty());
}

TEST(ConstEvaluator, MakeContiguousStridesZeroElement) {
    const std::vector<int64_t> shape{0, 3};
    const auto result = MakeContiguousStrides(shape);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->size(), 2U);
    EXPECT_EQ((*result)[1], 1);
    // strides[0] = 1 * 3 = 3 (no overflow since trailing dim is small)
    EXPECT_EQ((*result)[0], 3);
}

TEST(ConstEvaluator, MakeContiguousStridesOverflow) {
    constexpr int64_t kHuge = 1LL << 62;
    const std::vector<int64_t> shape{0, kHuge, kHuge};
    const auto result = MakeContiguousStrides(shape);
    ASSERT_FALSE(result.ok());
    EXPECT_EQ(result.status().code(), StatusCode::kResourceExhausted);
}

}// namespace
}// namespace aethermind
