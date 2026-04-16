#include "../../include/aethermind/execution/execution_plan.h"

#include <gtest/gtest.h>
#include <span>

namespace aethermind {
namespace {

struct TestAttrs {
    int epsilon;
    int axis;
};

Status FakeKernel() noexcept {
    return Status::Ok();
}

TEST(ExecutionPlan, AddStepCopiesAttrsIntoPlanOwnedStorage) {
    ExecutionPlan plan;
    TestAttrs attrs{.epsilon = 7, .axis = 3};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    ASSERT_TRUE(plan.AddStep(OperatorName("aethermind::rms_norm", ""),
                             ResolvedKernel{
                                     .op_type = OpType::kRMSNorm,
                                     .fn = &FakeKernel,
                                     .attrs = attrs_bytes,
                                     .debug_name = "test::fake_kernel",
                             })
                        .ok());

    attrs.epsilon = 99;

    ASSERT_EQ(plan.size(), 1U);
    ASSERT_EQ(plan.steps().front().kernel.attrs.size(), sizeof(TestAttrs));
    const auto* stored_attrs = reinterpret_cast<const TestAttrs*>(plan.steps().front().kernel.attrs.data());
    ASSERT_NE(stored_attrs, nullptr);
    EXPECT_NE(stored_attrs, &attrs);
    EXPECT_EQ(stored_attrs->epsilon, 7);
    EXPECT_EQ(stored_attrs->axis, 3);
    EXPECT_STREQ(plan.steps().front().kernel.debug_name, "test::fake_kernel");
}

TEST(ExecutionPlan, AddStepAllowsEmptyAttrSpan) {
    ExecutionPlan plan;

    const Status status = plan.AddStep(OperatorName("aethermind::rms_norm", ""),
                                       ResolvedKernel{
                                               .op_type = OpType::kRMSNorm,
                                               .fn = &FakeKernel,
                                               .attrs = {},
                                               .debug_name = "test::fake_kernel",
                                       });

    EXPECT_TRUE(status.ok());
    ASSERT_EQ(plan.size(), 1U);
    EXPECT_TRUE(plan.steps().front().kernel.attrs.empty());
}

}// namespace
}// namespace aethermind
