#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_constraint_evaluator.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/shape_inference/symbol_constraint_solver.h"

#include <gtest/gtest.h>

#include <vector>

namespace {
using namespace aethermind;

TEST(ShapeInferenceIncludes, ExposePublicTypes) {
    ShapeConstraint constraint;
    SymbolConstraintSolver solver;
    const ShapeSymbol symbol = ShapeSymbol::Create();
    std::vector<SymbolicShape> inputs{SymbolicShape(std::vector<ShapeSymbol>{symbol})};

    constraint.condition = DimEqualConstraint{};
    EXPECT_EQ(constraint.error_context, "");
    EXPECT_EQ(solver.EvaluateEqual(symbol, symbol), ShapeConstraintEvaluationResult::kSatisfied);

    const auto result = EvaluateShapeConstraint(
            constraint,
            std::span<const SymbolicShape>(inputs),
            std::span<const SymbolicShape>());
    EXPECT_EQ(result, ShapeConstraintEvaluationResult::kSatisfied);
}

}// namespace
