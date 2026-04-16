#include "aethermind/execution/executor.h"

#include <gtest/gtest.h>

#include <vector>

namespace aethermind {
namespace {

std::vector<int>* g_execution_order = nullptr;

Status FirstKernel() noexcept {
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(1);
    }
    return Status::Ok();
}

Status SecondKernel() noexcept {
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(2);
    }
    return Status::Ok();
}

Status FailingKernel() noexcept {
    return Status::InvalidArgument("kernel failure");
}

TEST(ExecutorBackendPath, ExecuteRunsFrozenKernelsInPlanOrder) {
    ExecutionPlan plan;
    std::vector<int> execution_order;
    g_execution_order = &execution_order;

    ASSERT_TRUE(plan.AddStep(ResolvedKernel{
                                     .op_type = OpType::kRMSNorm,
                                     .fn = &FirstKernel,
                                     .attrs = {},
                                     .debug_name = "test::first_kernel",
                             })
                        .ok());
    ASSERT_TRUE(plan.AddStep(ResolvedKernel{
                                     .op_type = OpType::kRoPE,
                                     .fn = &SecondKernel,
                                     .attrs = {},
                                     .debug_name = "test::second_kernel",
                             })
                        .ok());

    const Status status = Executor::Execute(plan);

    g_execution_order = nullptr;
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(execution_order, (std::vector<int>{1, 2}));
}

TEST(ExecutorBackendPath, ExecutePropagatesKernelFailure) {
    ExecutionPlan plan;
    ASSERT_TRUE(plan.AddStep(ResolvedKernel{
                                     .op_type = OpType::kRMSNorm,
                                     .fn = &FailingKernel,
                                     .attrs = {},
                                     .debug_name = "test::failing_kernel",
                             })
                        .ok());

    const Status status = Executor::Execute(plan);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
