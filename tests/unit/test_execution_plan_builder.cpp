#include "../../include/aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>
#include <span>

namespace aethermind {
namespace {

struct TestAttrs {
    int epsilon;
};

ExecutionPlanNodeSpec MakeRmsNormNodeSpec(std::span<const std::byte> attrs = {}) {
    return ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .attrs = attrs,
    };
}

TEST(ExecutionPlanBuilder, ResolveKernelForNodeUsesOpTypeDirectly) {
    CpuBackend backend;
    TestAttrs attrs{.epsilon = 42};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       MakeRmsNormNodeSpec(attrs_bytes));

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ(resolved->op_type, OpType::kRMSNorm);
    ASSERT_NE(resolved->fn, nullptr);
    EXPECT_EQ(resolved->attrs.data(), attrs_bytes.data());
    EXPECT_EQ(resolved->attrs.size(), sizeof(attrs));
    EXPECT_STREQ(resolved->debug_name, "test::fake_cpu_kernel");
}

TEST(ExecutionPlanBuilder, BuildFreezesResolvedKernelIntoExecutionPlan) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();
    TestAttrs attrs{.epsilon = 11};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(MakeRmsNormNodeSpec(attrs_bytes));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    attrs.epsilon = 99;

    const auto& step = plan->steps().front();
    ASSERT_EQ(step.kernel.attrs.size(), sizeof(TestAttrs));
    const auto* stored_attrs = reinterpret_cast<const TestAttrs*>(step.kernel.attrs.data());
    ASSERT_NE(stored_attrs, nullptr);
    EXPECT_EQ(step.kernel.op_type, OpType::kRMSNorm);
    EXPECT_EQ(stored_attrs->epsilon, 11);
    EXPECT_STREQ(step.kernel.debug_name, "test::fake_cpu_kernel");
}

TEST(ExecutionPlanBuilder, ResolveKernelForNodeRejectsUnknownOpType) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       ExecutionPlanNodeSpec{
                                                                .op_type = OpType::kUnknown,
                                                                .device_type = DeviceType::kCPU,
                                                                .activation_dtype = DataType::Float32(),
                                                                .weight_dtype = DataType::Float32(),
                                                        });

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
