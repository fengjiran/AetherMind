#include "../../include/aethermind/execution/execution_plan.h"

#include "aethermind/backend/kernel_context.h"
#include "aethermind/operators/function_operator.h"

#include <gtest/gtest.h>
#include <span>

namespace aethermind {
namespace {

struct TestAttrs {
    int epsilon;
    int axis;
};

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

TEST(ExecutionPlan, AddStepStoresOperatorResolvedAttrs) {
    ExecutionPlan plan;
    TestAttrs attrs{.epsilon = 7, .axis = 3};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});
    int packed_params = 17;

    ASSERT_TRUE(plan.AddStep(ExecutionStep{
                                     .op = std::make_shared<FunctionOperator>(
                                             OpType::kRmsNorm,
                                             &FakeKernel,
                                             attrs_bytes,
                                             "test::fake_kernel"),
                                     .packed_params = &packed_params,
                                     .workspace_requirement = {
                                             .bytes = 128,
                                             .alignment = 64,
                                             .offset = 256,
                                     },
                                     .debug_name = "test::fake_kernel",
                             })
                        .ok());

    attrs.epsilon = 99;

    ASSERT_EQ(plan.size(), 1U);
    const auto& step = plan.steps().front();
    const ResolvedKernel resolved = step.op->GetResolvedKernel();
    ASSERT_EQ(resolved.attrs.size(), sizeof(TestAttrs));
    const auto* stored_attrs = reinterpret_cast<const TestAttrs*>(resolved.attrs.data());
    ASSERT_NE(stored_attrs, nullptr);
    EXPECT_NE(stored_attrs, &attrs);
    EXPECT_EQ(step.op->Type(), OpType::kRmsNorm);
    EXPECT_EQ(step.selector.device_type, DeviceType::kUndefined);
    EXPECT_EQ(step.packed_params, &packed_params);
    EXPECT_EQ(step.workspace_requirement.bytes, 128U);
    EXPECT_EQ(step.workspace_requirement.alignment, 64U);
    EXPECT_EQ(step.workspace_requirement.offset, 256U);
    EXPECT_EQ(stored_attrs->epsilon, 7);
    EXPECT_EQ(stored_attrs->axis, 3);
    EXPECT_STREQ(step.debug_name, "test::fake_kernel");
}

TEST(ExecutionPlan, AddStepAllowsEmptyAttrSpan) {
    ExecutionPlan plan;

    const Status status = plan.AddStep(ExecutionStep{
            .op = std::make_shared<FunctionOperator>(
                    OpType::kRmsNorm,
                    &FakeKernel,
                    std::span<const std::byte>{},
                    "test::fake_kernel"),
            .debug_name = "test::fake_kernel",
    });

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(plan.size(), 1U);
    EXPECT_TRUE(plan.steps().front().op->GetResolvedKernel().attrs.empty());
    EXPECT_EQ(plan.steps().front().packed_params, nullptr);
}

TEST(ExecutionPlan, AddStepRejectsInvalidWorkspaceAlignment) {
    ExecutionPlan plan;

    const Status status = plan.AddStep(ExecutionStep{
            .op = std::make_shared<FunctionOperator>(
                    OpType::kRmsNorm,
                    &FakeKernel,
                    std::span<const std::byte>{},
                    "test::fake_kernel"),
            .workspace_requirement = {
                    .bytes = 64,
                    .alignment = 24,
                    .offset = 0,
            },
            .debug_name = "test::fake_kernel",
    });

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
