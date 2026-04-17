#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

KernelSelector MakeCpuSelector(ExecPhase phase = ExecPhase::kBoth,
                               IsaLevel isa = IsaLevel::kScalar) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = isa,
            .phase = phase,
    };
}

TEST(CpuResolveKernel, RegisteredKeyReturnsKernel) {
    CpuBackend backend;

    EXPECT_NE(backend.ResolveKernel(OpType::kRMSNorm, MakeCpuSelector()), nullptr);
}

TEST(CpuResolveKernel, MissingKeyReturnsNullptr) {
    CpuBackend backend;

    EXPECT_EQ(backend.ResolveKernel(OpType::kLinear, MakeCpuSelector()), nullptr);
}

TEST(CpuResolveKernel, DebugRegistryIsExposedForInspection) {
    CpuBackend backend;
    EXPECT_NE(backend.TryGetKernelRegistryForDebug(), nullptr);
}

TEST(CpuResolveKernel, RegisteredKernelCanBeInvoked) {
    CpuBackend backend;

    const KernelFunc fn = backend.ResolveKernel(OpType::kRMSNorm, MakeCpuSelector());
    ASSERT_NE(fn, nullptr);

    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const CpuRmsNormParams params{
            .Input = input,
            .Weight = weight,
            .Output = output,
            .HiddenSize = 4,
    };
    const CpuRmsNormAttrs attrs{.Epsilon = 1.0e-5F};

    const Status status = fn(KernelInvocation{.op_type = OpType::kRMSNorm,
                                              .selector = MakeCpuSelector()},
                             OpKernelContext{
                                     .device = Device::CPU(),
                                     .packed_params = &params,
                                     .attrs = std::as_bytes(std::span{&attrs, size_t{1}}),
                             },
                             WorkspaceBinding{});
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
}

}// namespace
