#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/kernel_invoker.h"

#include <gtest/gtest.h>

#include <span>

namespace {

using namespace aethermind;

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

int g_kernel_invocations = 0;
int g_built_param = 0;

Status CaptureBuiltParamKernel(const KernelContext& context) noexcept {
    ++g_kernel_invocations;
    if (context.kernel_params == nullptr) {
        return Status::FailedPrecondition("Expected params from the builder");
    }
    g_built_param = *static_cast<const int*>(context.kernel_params);
    return Status::Ok();
}

Status BuildKernelParam(std::span<const TensorView>,
                        std::span<const MutableTensorView>,
                        void* params_buffer) noexcept {
    *static_cast<int*>(params_buffer) = 17;
    return Status::Ok();
}

Status FailToBuildKernelParam(std::span<const TensorView>,
                              std::span<const MutableTensorView>,
                              void*) noexcept {
    return Status::InvalidArgument("test params builder failure");
}

TEST(ExecutionPlan, StoresResolvedKernelByValue) {
    ResolvedKernel kernel{
            .op_type = OpType::kSoftmax,
            .fn = &FakeKernel,
            .attrs = {std::byte{7}},
            .debug_name = "test::fake_kernel",
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlan::Create({
            ExecutionStep{
                    .selector = KernelSelector{.device_type = DeviceType::kCPU},
                    .kernel = kernel,
                    .workspace_requirement = {.bytes = 128, .alignment = 64},
            },
    });

    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    const ExecutionStep& step = plan->steps().front();
    EXPECT_EQ(step.kernel.op_type, OpType::kSoftmax);
    EXPECT_EQ(step.kernel.fn, &FakeKernel);
    EXPECT_EQ(step.kernel.attrs, kernel.attrs);
    EXPECT_NE(step.kernel.attrs.data(), kernel.attrs.data());
    EXPECT_STREQ(step.kernel.debug_name, "test::fake_kernel");
    EXPECT_EQ(step.workspace_requirement.bytes, 128U);
}

TEST(ExecutionPlan, RejectsInvalidResolvedKernel) {
    EXPECT_EQ(ExecutionPlan::Create({ExecutionStep{}}).status().code(),
              StatusCode::kInvalidArgument);

    EXPECT_EQ(ExecutionPlan::Create({ExecutionStep{
                                            .kernel = ResolvedKernel{
                                                    .op_type = OpType::kSoftmax,
                                                    .fn = &FakeKernel,
                                                    .params_size = 1,
                                            },
                                    }})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(KernelInvoker, BuildsParamsOnlyForTheKernelCall) {
    g_kernel_invocations = 0;
    g_built_param = 0;
    KernelContext context{};
    const Status status = InvokeKernel(
            ResolvedKernel{
                    .op_type = OpType::kAdd,
                    .fn = &CaptureBuiltParamKernel,
                    .params_builder = &BuildKernelParam,
                    .params_size = sizeof(int),
            },
            context, {}, {});

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(g_kernel_invocations, 1);
    EXPECT_EQ(g_built_param, 17);
}

TEST(KernelInvoker, PropagatesBuilderFailureWithoutCallingKernel) {
    g_kernel_invocations = 0;
    KernelContext context{};
    const Status status = InvokeKernel(
            ResolvedKernel{
                    .op_type = OpType::kAdd,
                    .fn = &CaptureBuiltParamKernel,
                    .params_builder = &FailToBuildKernelParam,
                    .params_size = sizeof(int),
            },
            context, {}, {});

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(g_kernel_invocations, 0);
}

TEST(KernelInvoker, RejectsNullFunctionAndInvalidParamsContract) {
    KernelContext context{};
    EXPECT_EQ(InvokeKernel(ResolvedKernel{.op_type = OpType::kAdd}, context, {}, {})
                      .code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(InvokeKernel(ResolvedKernel{
                                   .op_type = OpType::kAdd,
                                   .fn = &FakeKernel,
                                   .params_size = 1,
                           },
                           context, {}, {})
                      .code(),
              StatusCode::kFailedPrecondition);
}

}// namespace
