#include "test_const_eval_helpers.h"

#include <gtest/gtest.h>

#include "aethermind/operators/op_params.h"

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

using namespace aethermind;

// A size-1 axis may carry any non-negative stride. INT64_MAX makes an
// unconditional carry advance overflow once the input offset is nonzero, so
// these cases pin the shared strided kernels to skip singleton axes and to
// stop advancing after the final element.
constexpr std::array<int64_t, 2> kShape{2, 1};
constexpr std::array<int64_t, 2> kStridedInputStrides{
        kShape[0], std::numeric_limits<int64_t>::max()};
constexpr std::array<int64_t, 2> kContiguousOutputStrides{1, 1};

TEST(ConstEvalStridedSingletonStride, CastReadsElementsAcrossSingletonAxisStride) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kCast);
    ASSERT_NE(evaluator, nullptr);

    constexpr std::array<float, 3> input_values{1.5F, 0.0F, 2.5F};
    std::array<BFloat16, 2> output_values{};
    const std::vector<TensorView> inputs{
            TensorView(input_values.data(), DataType::Float32(), kShape,
                       kStridedInputStrides)};
    std::vector<MutableTensorView> outputs{
            MutableTensorView(output_values.data(), DataType::BFloat(16), kShape,
                              kContiguousOutputStrides)};

    const Status status = evaluator->Evaluate(
            inputs, outputs, CastParams{.target_dtype = DataType::BFloat(16)});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(static_cast<float>(output_values[0]), 1.5F);
    EXPECT_EQ(static_cast<float>(output_values[1]), 2.5F);
}

TEST(ConstEvalStridedSingletonStride, AddReadsElementsAcrossSingletonAxisStride) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kAdd);
    ASSERT_NE(evaluator, nullptr);

    constexpr std::array<float, 3> lhs_values{1.0F, 0.0F, 2.0F};
    constexpr std::array<float, 3> rhs_values{10.0F, 0.0F, 20.0F};
    std::array<float, 2> output_values{};
    const std::vector<TensorView> inputs{
            TensorView(lhs_values.data(), DataType::Float32(), kShape,
                       kStridedInputStrides),
            TensorView(rhs_values.data(), DataType::Float32(), kShape,
                       kStridedInputStrides)};
    std::vector<MutableTensorView> outputs{
            MutableTensorView(output_values.data(), DataType::Float32(), kShape,
                              kContiguousOutputStrides)};

    const Status status = evaluator->Evaluate(inputs, outputs, AddParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output_values[0], 11.0F);
    EXPECT_FLOAT_EQ(output_values[1], 22.0F);
}

TEST(ConstEvalStridedSingletonStride, SiluReadsElementsAcrossSingletonAxisStride) {
    const ConstEvaluator* evaluator = FindConstEvaluator(OpType::kSilu);
    ASSERT_NE(evaluator, nullptr);

    constexpr std::array<float, 3> input_values{1.0F, 0.0F, 2.0F};
    std::array<float, 2> output_values{};
    const std::vector<TensorView> inputs{
            TensorView(input_values.data(), DataType::Float32(), kShape,
                       kStridedInputStrides)};
    std::vector<MutableTensorView> outputs{
            MutableTensorView(output_values.data(), DataType::Float32(), kShape,
                              kContiguousOutputStrides)};

    const Status status = evaluator->Evaluate(inputs, outputs, SiluParams{});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output_values[0], 0.7310586F, 1.0e-5F);
    EXPECT_NEAR(output_values[1], 1.7615942F, 1.0e-5F);
}

} // namespace