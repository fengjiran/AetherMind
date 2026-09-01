#include "../test_operator_inference_helpers.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_inference.h"

#include <gtest/gtest.h>

#include <limits>

namespace {
using namespace aethermind;

GateUpLinearParams MakeParams(int64_t gate = 16, int64_t up = 16) {
    return GateUpLinearParams{
            .gate_out_features = gate,
            .up_out_features = up,
            .has_bias = false,
    };
}

TEST(GateUpLinearInference, InfersOrderedGateAndUpOutputs) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 4096}),
            MakeSpec(DataType::Float32(), {24, 4096}),
    };

    const StatusOr<InferenceResult> result =
            InferOperator(OpType::kGateUpLinear, MakeParams(16, 8), inputs);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 2U);
    EXPECT_EQ(result->outputs[0], MakeSpec(DataType::Float32(), {4, 16}));
    EXPECT_EQ(result->outputs[1], MakeSpec(DataType::Float32(), {4, 8}));
    EXPECT_TRUE(result->runtime_checks.empty());
}

TEST(GateUpLinearInference, RejectsUnsupportedParamsAndPackedWeightContracts) {
    const TensorSpec valid_inputs[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {16, 8}),
    };

    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear, AddParams{}, valid_inputs).ok());
    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear,
                               GateUpLinearParams{.gate_out_features = 8,
                                                  .up_out_features = 8,
                                                  .has_bias = true},
                               valid_inputs)
                         .ok());
    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear, MakeParams(-1, 8), valid_inputs).ok());
    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear,
                               MakeParams(std::numeric_limits<int64_t>::max(), 1),
                               valid_inputs)
                         .ok());

    const TensorSpec wrong_rows[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {15, 8}),
    };
    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear, MakeParams(8, 8), wrong_rows).ok());

    const TensorSpec wrong_k[2] = {
            MakeSpec(DataType::Float32(), {4, 8}),
            MakeSpec(DataType::Float32(), {16, 7}),
    };
    EXPECT_FALSE(InferOperator(OpType::kGateUpLinear, MakeParams(8, 8), wrong_k).ok());
}

TEST(GateUpLinearInference, DefersDistinctSymbolicInputFeatures) {
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_k = ShapeSymbol::Create();
    const ShapeSymbol weight_k = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, input_k})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{
                     ShapeSymbol::CreateFromValue(16), weight_k})},
    };

    const StatusOr<InferenceResult> result =
            InferOperator(OpType::kGateUpLinear, MakeParams(8, 8), inputs);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->runtime_checks.size(), 1U);
    EXPECT_TRUE(std::holds_alternative<DimEqualConstraint>(result->runtime_checks[0].condition));
}

TEST(GateUpLinearInference, AcceptsQuantizedWeightAndRankOneInput) {
    const TensorSpec inputs[2] = {
            MakeSpec(DataType::BFloat(16), {8}),
            MakeSpec(DataType::Int(4), {16, 8}),
    };

    const StatusOr<InferenceResult> result =
            InferOperator(OpType::kGateUpLinear, MakeParams(8, 8), inputs);

    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->outputs.size(), 2U);
    EXPECT_EQ(result->outputs[0], MakeSpec(DataType::BFloat(16), {8}));
    EXPECT_EQ(result->outputs[1], MakeSpec(DataType::BFloat(16), {8}));
}

} // namespace
