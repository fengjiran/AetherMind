#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace {

using namespace aethermind;

KernelSelector MakeCpuSelector(ExecPhase phase = ExecPhase::kBoth) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = phase,
    };
}

TEST(CpuPrepareKernel, RegisteredKeyReturnsPreparedKernel) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved = backend.PrepareKernel(
            OpType::kRmsNorm, MakeCpuSelector(), OpParams{RmsNormParams{.eps = 1.0e-5F}});

    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_NE(resolved->fn, nullptr);
    EXPECT_EQ(resolved->op_type, OpType::kRmsNorm);
    EXPECT_EQ(resolved->attrs.size(), sizeof(float));
    EXPECT_TRUE(resolved->workspace_requirement.empty());
    EXPECT_EQ(resolved->workspace_requirement.alignment, 64U);
}

TEST(CpuPrepareKernel, MissingKeyReturnsNotFound) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved = backend.PrepareKernel(
            OpType::kLinear, MakeCpuSelector(), OpParams{LinearParams{}});

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

TEST(CpuPrepareKernel, RejectsInvalidKernelMetadata) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved = backend.PrepareKernel(
            OpType::kRmsNorm, MakeCpuSelector(), OpParams{RmsNormParams{.eps = 0.0F}});

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

TEST(CpuPrepareKernel, DebugRegistryIsExposedForInspection) {
    CpuBackend backend;
    EXPECT_NE(backend.TryGetKernelRegistryForDebug(), nullptr);
}

TEST(CpuPrepareKernel, PreparedKernelCanBeInvoked) {
    CpuBackend backend;
    const StatusOr<ResolvedKernel> resolved = backend.PrepareKernel(
            OpType::kRmsNorm, MakeCpuSelector(), OpParams{RmsNormParams{.eps = 1.0e-5F}});
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();

    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t io_shape[2] = {1, 4};
    const int64_t io_strides[2] = {4, 1};
    const int64_t w_shape[1] = {4};
    const int64_t w_strides[1] = {1};

    // The params builder prepares compute-ready args once, mirroring what
    // PrepareExecutionBindings does for a PreparedExecutionBindings-owned arena slot.
    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
    const std::array<TensorView, 2> inputs{
            TensorView{input, DataType::Float32(), io_shape, io_strides},
            TensorView{weight, DataType::Float32(), w_shape, w_strides},
    };
    const std::array<MutableTensorView, 1> outputs{
            MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const Status build_status = resolved->params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = resolved->attrs,
            },
            storage.data());
    ASSERT_TRUE(build_status.ok()) << build_status.ToString();

    const Status status = resolved->fn(KernelContext{
            .workspace_binding = {},
            .kernel_params = storage.data(),
            .attrs = resolved->attrs,
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
}

} // namespace
