#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

TEST(GateUpLinearOp, RegistersOperatorFactory) {
    const StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kGateUpLinear,
            GateUpLinearParams{.gate_out_features = 8,
                               .up_out_features = 8,
                               .has_bias = false});

    ASSERT_TRUE(op.ok()) << op.status().ToString();
    EXPECT_EQ((*op)->Type(), OpType::kGateUpLinear);
}

}// namespace
