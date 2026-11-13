#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// Cast is a storage-format transition across the frozen 16-bit formats; the
// output shape is always identical to the input shape.

TEST(CastInference, InfersEverySupportedDirection) {
    struct Case {
        DataType source;
        DataType target;
    };
    const Case cases[] = {
            {DataType::Float32(), DataType::Float(16)},
            {DataType::Float32(), DataType::BFloat(16)},
            {DataType::Float(16), DataType::Float32()},
            {DataType::Float(16), DataType::BFloat(16)},
            {DataType::BFloat(16), DataType::Float32()},
            {DataType::BFloat(16), DataType::Float(16)},
    };
    for (const Case& test_case: cases) {
        const TensorSpec inputs[1] = {MakeSpec(test_case.source, {3, 5})};
        const auto result = InferOperator(
                OpType::kCast,
                OpParams{CastParams{.target_dtype = test_case.target}}, inputs);
        ASSERT_TRUE(result.ok()) << result.status().ToString();
        ASSERT_EQ(result->outputs.size(), 1U);
        EXPECT_EQ(result->outputs[0].dtype, test_case.target);
        EXPECT_EQ(result->outputs[0].shape, inputs[0].shape);
        EXPECT_TRUE(result->runtime_checks.empty());
    }
}

TEST(CastInference, PreservesSymbolicShape) {
    const ShapeSymbol rows = ShapeSymbol::Create();
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::BFloat(16),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{rows, ShapeSymbol::CreateFromValue(4)})},
    };
    const auto result = InferOperator(
            OpType::kCast,
            OpParams{CastParams{.target_dtype = DataType::Float32()}}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(result->outputs[0].shape[0], rows);
}

TEST(CastInference, RejectsSameDtype) {
    const TensorSpec inputs[1] = {MakeSpec(DataType::BFloat(16), {2, 2})};
    const auto result = InferOperator(
            OpType::kCast,
            OpParams{CastParams{.target_dtype = DataType::BFloat(16)}}, inputs);
    EXPECT_EQ(result.status(), StatusCode::kInvalidArgument);
}

TEST(CastInference, RejectsUndefinedTarget) {
    const TensorSpec inputs[1] = {MakeSpec(DataType::Float32(), {2, 2})};
    const auto result = InferOperator(
            OpType::kCast, OpParams{CastParams{}}, inputs);
    EXPECT_EQ(result.status(), StatusCode::kInvalidArgument);
}

TEST(CastInference, RejectsUnsupportedSourceAndTarget) {
    const TensorSpec int_inputs[1] = {MakeSpec(DataType::Int(32), {2, 2})};
    EXPECT_EQ(InferOperator(OpType::kCast,
                            OpParams{CastParams{.target_dtype = DataType::Float32()}},
                            int_inputs)
                      .status(),
              StatusCode::kInvalidArgument);

    const TensorSpec float_inputs[1] = {MakeSpec(DataType::Float32(), {2, 2})};
    EXPECT_EQ(InferOperator(OpType::kCast,
                            OpParams{CastParams{.target_dtype = DataType::Double()}},
                            float_inputs)
                      .status(),
              StatusCode::kInvalidArgument);
}

TEST(CastInference, RejectsWrongArity) {
    const TensorSpec inputs[2] = {MakeSpec(DataType::Float32(), {2}),
                                  MakeSpec(DataType::Float32(), {2})};
    EXPECT_EQ(InferOperator(OpType::kCast,
                            OpParams{CastParams{.target_dtype = DataType::BFloat(16)}},
                            inputs)
                      .status(),
              StatusCode::kInvalidArgument);
}

} // namespace
