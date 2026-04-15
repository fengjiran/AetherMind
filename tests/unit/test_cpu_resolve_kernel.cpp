#include "aethermind/backend/cpu/cpu_backend.h"

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

    const Status status = fn();
    EXPECT_TRUE(status.ok());
}

}// namespace
