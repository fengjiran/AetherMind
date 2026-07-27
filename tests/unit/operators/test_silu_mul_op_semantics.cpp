#include "test_operator_semantics_helpers.h"

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>
#include <vector>

namespace {
using namespace aethermind;

TEST(OperatorSemanticsValidate, SiluMulValidParams) {
    auto gate = MakeSpec(DataType::Float32(), {4, 256});
    auto up = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    EXPECT_TRUE(InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs).ok());
}

TEST(OperatorSemanticsInfer, SiluMulFloat32Ok) {
    auto gate = MakeSpec(DataType::Float32(), {4, 256});
    auto up = MakeSpec(DataType::Float32(), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::Float32());
}

TEST(OperatorSemanticsInfer, SiluMulBFloat16Ok) {
    auto gate = MakeSpec(DataType::BFloat(16), {4, 256});
    auto up = MakeSpec(DataType::BFloat(16), {4, 256});
    std::vector<TensorSpec> inputs = {gate, up};
    auto result = InferOperator(OpType::kSiluMul, SiluMulParams{}, inputs);
    ASSERT_TRUE(result.ok());
    EXPECT_EQ(result->outputs[0].dtype, DataType::BFloat(16));
}

}// namespace
