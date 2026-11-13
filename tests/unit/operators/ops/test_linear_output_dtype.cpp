#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// Linear output_dtype carries the mixed-output contract used by an A16 LMHead
// that must emit FP32 logits without narrowing them first.

TEST(LinearOutputDType, DefaultOutputFollowsActivation) {
    const TensorSpec inputs[2] = {MakeSpec(DataType::BFloat(16), {2, 4}),
                                  MakeSpec(DataType::BFloat(16), {8, 4})};
    const auto result = InferOperator(OpType::kLinear, OpParams{LinearParams{}}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

TEST(LinearOutputDType, ExplicitFloat32WinsOverActivation) {
    const TensorSpec inputs[2] = {MakeSpec(DataType::BFloat(16), {2, 4}),
                                  MakeSpec(DataType::BFloat(16), {8, 4})};
    const auto result = InferOperator(
            OpType::kLinear,
            OpParams{LinearParams{.output_dtype = DataType::Float32()}}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(result->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(result->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(LinearOutputDType, AcceptsFloat16Output) {
    const TensorSpec inputs[2] = {MakeSpec(DataType::Float32(), {1, 4}),
                                  MakeSpec(DataType::Float(16), {8, 4})};
    const auto result = InferOperator(
            OpType::kLinear,
            OpParams{LinearParams{.output_dtype = DataType::Float(16)}}, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float(16));
}

TEST(LinearOutputDType, RejectsUnsupportedExplicitOutput) {
    const TensorSpec inputs[2] = {MakeSpec(DataType::Float32(), {1, 4}),
                                  MakeSpec(DataType::Float32(), {8, 4})};
    EXPECT_EQ(InferOperator(OpType::kLinear,
                            OpParams{LinearParams{.output_dtype = DataType::Double()}},
                            inputs)
                      .status(),
              StatusCode::kInvalidArgument);
}

} // namespace
