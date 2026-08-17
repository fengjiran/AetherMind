#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/base/device.h"
#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

KernelSelector MakeCpuSelector(ExecPhase phase = ExecPhase::kBoth) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = phase,
    };
}

TEST(CpuBackend, DeviceTypeIsCPU) {
    CpuBackend backend;
    EXPECT_EQ(backend.device_type(), DeviceType::kCPU);
}

TEST(CpuBackend, CapabilitiesExposeCPUType) {
    CpuBackend backend;
    const auto& caps = backend.capabilities();
    EXPECT_EQ(caps.device_type, DeviceType::kCPU);
}

TEST(CpuBackend, PrepareKernelRejectsMissingDescriptor) {
    CpuBackend backend;
    const StatusOr<ResolvedKernel> resolved = backend.PrepareKernel(
            OpType::kLinear, MakeCpuSelector(), OpParams{LinearParams{}});
    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

TEST(CpuBackend, TryGetKernelRegistryForDebugReturnsRegistry) {
    CpuBackend backend;
    EXPECT_NE(backend.TryGetKernelRegistryForDebug(), nullptr);
}

TEST(CpuBackendFactory, CreatesValidBackend) {
    CpuBackendFactory factory;
    EXPECT_EQ(factory.device_type(), DeviceType::kCPU);

    auto backend = factory.Create();
    ASSERT_NE(backend, nullptr);
    EXPECT_EQ(backend->device_type(), DeviceType::kCPU);
}

TEST(CpuBackend, RuntimeBuilderDefaultProvidesCpuBackend) {
    RuntimeBuilder builder;
    RuntimeContext context = builder.Build();

    auto backend_or = context.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(backend_or.ok());
    ASSERT_NE(backend_or.value(), nullptr);
    EXPECT_EQ(backend_or.value()->device_type(), DeviceType::kCPU);
}

}// namespace
