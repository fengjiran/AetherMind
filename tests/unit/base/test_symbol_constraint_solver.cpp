#include "aethermind/base/symbol_constraint_solver.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

TEST(SymbolConstraintSolver, AddsTransitiveSymbolEquality) {
    SymbolConstraintSolver solver;
    const ShapeSymbol a = ShapeSymbol::Create();
    const ShapeSymbol b = ShapeSymbol::Create();
    const ShapeSymbol c = ShapeSymbol::Create();

    ASSERT_TRUE(solver.AddEqual(a, b).ok());
    ASSERT_TRUE(solver.AddEqual(b, c).ok());

    EXPECT_TRUE(solver.AreEqual(a, c));
    EXPECT_EQ(solver.EvaluateEqual(a, c), ShapeConstraintEvaluationResult::kSatisfied);
}

TEST(SymbolConstraintSolver, PropagatesStaticBindingThroughClass) {
    SymbolConstraintSolver solver;
    const ShapeSymbol a = ShapeSymbol::Create();
    const ShapeSymbol b = ShapeSymbol::Create();

    ASSERT_TRUE(solver.AddEqual(a, ShapeSymbol::CreateFromValue(4096)).ok());
    ASSERT_TRUE(solver.AddEqual(a, b).ok());

    ASSERT_TRUE(solver.GetStaticBinding(b).has_value());
    EXPECT_EQ(*solver.GetStaticBinding(b), 4096);
    EXPECT_EQ(solver.EvaluateEqual(b, ShapeSymbol::CreateFromValue(4096)),
              ShapeConstraintEvaluationResult::kSatisfied);
}

TEST(SymbolConstraintSolver, RejectsStaticConflicts) {
    SymbolConstraintSolver solver;
    const ShapeSymbol a = ShapeSymbol::Create();
    const ShapeSymbol b = ShapeSymbol::Create();

    ASSERT_TRUE(solver.AddEqual(a, 8).ok());
    EXPECT_EQ(solver.AddEqual(a, 16).code(), StatusCode::kInvalidArgument);

    ASSERT_TRUE(solver.AddEqual(b, 16).ok());
    EXPECT_EQ(solver.AddEqual(a, b).code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(solver.AddEqual(ShapeSymbol::CreateFromValue(8), ShapeSymbol::CreateFromValue(16)).code(),
              StatusCode::kInvalidArgument);
}

TEST(SymbolConstraintSolver, TreatsUnknownAsNonPropagating) {
    SymbolConstraintSolver solver;
    const ShapeSymbol unknown = ShapeSymbol::Unknown();
    const ShapeSymbol symbol = ShapeSymbol::Create();

    ASSERT_TRUE(solver.AddEqual(unknown, symbol).ok());
    ASSERT_TRUE(solver.AddEqual(unknown, ShapeSymbol::CreateFromValue(8)).ok());

    EXPECT_EQ(solver.EvaluateEqual(unknown, unknown), ShapeConstraintEvaluationResult::kDeferred);
    EXPECT_EQ(solver.EvaluateEqual(unknown, symbol), ShapeConstraintEvaluationResult::kDeferred);
    EXPECT_FALSE(solver.GetStaticBinding(unknown).has_value());
    EXPECT_FALSE(solver.GetStaticBinding(symbol).has_value());
}

TEST(SymbolConstraintSolver, EvaluatesSeparateClassesWithSameStaticBindingAsEqual) {
    SymbolConstraintSolver solver;
    const ShapeSymbol a = ShapeSymbol::Create();
    const ShapeSymbol b = ShapeSymbol::Create();

    ASSERT_TRUE(solver.AddEqual(a, 8).ok());
    ASSERT_TRUE(solver.AddEqual(b, 8).ok());

    EXPECT_EQ(solver.EvaluateEqual(a, b), ShapeConstraintEvaluationResult::kSatisfied);
}

TEST(SymbolConstraintSolver, DefersUnrelatedSymbols) {
    SymbolConstraintSolver solver;
    const ShapeSymbol a = ShapeSymbol::Create();
    const ShapeSymbol b = ShapeSymbol::Create();

    EXPECT_EQ(solver.EvaluateEqual(a, b), ShapeConstraintEvaluationResult::kDeferred);
    EXPECT_EQ(solver.EvaluateEqual(a, ShapeSymbol::CreateFromValue(8)), ShapeConstraintEvaluationResult::kDeferred);
}

}// namespace
