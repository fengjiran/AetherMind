#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/runtime/workspace.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include "aethermind/dtypes/data_type.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

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
    requires requires(cpu::detail::RmsNormParams params, TensorView input_view, TensorView weight_view,
                      MutableTensorView output_view) {
        params.input_tensor = input_view;
        params.weight_tensor = weight_view;
        params.output_tensor = output_view;
    }
cpu::detail::RmsNormParams MakeRmsNormParams(const float (&input)[N],
                                           const float (&weight)[N],
                                           float (&output)[N],
                                           const int64_t (&shape)[1],
                                           const int64_t (&strides)[1]) {
    const TensorView input_view = MakeContiguousFloatTensorView(input, shape, strides);
    const TensorView weight_view = MakeContiguousFloatTensorView(weight, shape, strides);
    const MutableTensorView output_view = MakeContiguousMutableFloatTensorView(output, shape, strides);

    return cpu::detail::RmsNormParams{
            .input_tensor = input_view,
            .weight_tensor = weight_view,
            .output_tensor = output_view,
    };
}

KernelSelector MakeCpuSelector(ExecPhase phase = ExecPhase::kBoth,
                               IsaLevel isa = IsaLevel::kScalar) {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = isa,
            .phase = phase,
    };
}

TEST(CpuResolveKernel, RegisteredKeyReturnsKernel) {
    CpuBackend backend;

    EXPECT_NE(backend.ResolveKernel(OpType::kRmsNorm, MakeCpuSelector()), nullptr);
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

    const KernelFunc fn = backend.ResolveKernel(OpType::kRmsNorm, MakeCpuSelector());
    ASSERT_NE(fn, nullptr);

    const float input[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    const float weight[4] = {1.0F, 1.0F, 1.0F, 1.0F};
    float output[4] = {0.0F, 0.0F, 0.0F, 0.0F};
    const int64_t io_shape[2] = {1, 4};
    const int64_t io_strides[2] = {4, 1};
    const int64_t w_shape[1] = {4};
    const int64_t w_strides[1] = {1};
    const cpu::detail::RmsNormParams params{
            .input_tensor = TensorView{input, DataType::Float32(), io_shape, io_strides},
            .weight_tensor = TensorView{weight, DataType::Float32(), w_shape, w_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), io_shape, io_strides},
    };
    const float epsilon = 1.0e-5F;

    const Status status = fn(KernelContext{
            .workspace_binding = {},
            .kernel_params = &params,
            .attrs = std::as_bytes(std::span{&epsilon, size_t{1}}),
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
    EXPECT_NEAR(output[0], 0.365148, 1e-5);
}

}// namespace
