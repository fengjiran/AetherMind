#include "../../include/aethermind/execution/workspace_types.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/cpu/kernels/cpu_rmsnorm_kernel.h"
#include "aethermind/backend/kernel_invocation.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/base/tensor_view.h"

#include "data_type.h"

#include <gtest/gtest.h>

using namespace aethermind;

namespace {

template<size_t N>
TensorView MakeContiguousFloatTensorView(const float (&data)[N],
                                         const int64_t (&shape)[1],
                                         const int64_t (&strides)[1]) {
    return TensorView{data, DataType::Float32(), shape, strides};
}

template<size_t N>
MutableTensorView MakeContiguousMutableFloatTensorView(float (&data)[N],
                                                       const int64_t (&shape)[1],
                                                       const int64_t (&strides)[1]) {
    return MutableTensorView{data, DataType::Float32(), shape, strides};
}

template<size_t N>
    requires requires(CpuRmsNormParams params, TensorView input_view, TensorView weight_view,
                      MutableTensorView output_view) {
        params.Input = input_view;
        params.Weight = weight_view;
        params.Output = output_view;
    }
CpuRmsNormParams MakeCpuRmsNormParams(const float (&input)[N],
                                      const float (&weight)[N],
                                      float (&output)[N],
                                      const int64_t (&shape)[1],
                                      const int64_t (&strides)[1]) {
    const TensorView input_view = MakeContiguousFloatTensorView(input, shape, strides);
    const TensorView weight_view = MakeContiguousFloatTensorView(weight, shape, strides);
    const MutableTensorView output_view = MakeContiguousMutableFloatTensorView(output, shape, strides);

    return CpuRmsNormParams{
            .Input = input_view,
            .Weight = weight_view,
            .Output = output_view,
    };
}

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
    const int64_t shape[1] = {4};
    const int64_t strides[1] = {1};
    const CpuRmsNormParams params = MakeCpuRmsNormParams(input, weight, output, shape, strides);
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
