#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_params.h"
#include "backend/cpu/kernels/elementwise_mul/elementwise_mul_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>

namespace {

using namespace aethermind;

KernelSelector MakeSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

StatusOr<ResolvedKernel> PrepareKernel() {
    CpuBackend backend;
    return backend.PrepareKernel(
            OpType::kElementwiseMul, MakeSelector(),
            OpParams{ElementwiseMulParams{}});
}

/// Test-side views mirroring the binding-time inputs of the builder.
struct ElementwiseMulTestViews {
    TensorView lhs_tensor{};
    TensorView rhs_tensor{};
    MutableTensorView output_tensor{};
};

/// Builds the prepared args through the registered builder (the cold path
/// BuildExecutionBindings uses) and then runs the thin entry.
Status RunElementwiseMul(const ElementwiseMulTestViews& views) noexcept {
    const StatusOr<ResolvedKernel> kernel = PrepareKernel();
    if (!kernel.ok()) {
        return kernel.status();
    }

    alignas(std::max_align_t) std::array<std::byte, kMaxKernelParamsSize> storage{};
    const std::array<TensorView, 2> inputs{views.lhs_tensor, views.rhs_tensor};
    const std::array<MutableTensorView, 1> outputs{views.output_tensor};
    const Status build_status = kernel->params_builder(
            KernelParamsBuildContext{
                    .inputs = inputs,
                    .outputs = outputs,
                    .attrs = kernel->attrs,
            },
            storage.data());
    if (!build_status.ok()) {
        return build_status;
    }
    return kernel->fn(KernelContext{
            .kernel_params = storage.data(),
            .attrs = kernel->attrs,
    });
}

TEST(ElementwiseMulKernel, SameShapeContiguous) {
    constexpr float lhs[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    constexpr float rhs[6] = {10.0F, 20.0F, 30.0F, 40.0F, 50.0F, 60.0F};
    float output[6] = {};
    constexpr int64_t shape[2] = {2, 3};
    constexpr int64_t strides[2] = {3, 1};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{lhs, DataType::Float32(), shape, strides},
            .rhs_tensor = TensorView{rhs, DataType::Float32(), shape, strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), shape, strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    EXPECT_FLOAT_EQ(output[0], 10.0F);
    EXPECT_FLOAT_EQ(output[1], 40.0F);
    EXPECT_FLOAT_EQ(output[2], 90.0F);
    EXPECT_FLOAT_EQ(output[3], 160.0F);
    EXPECT_FLOAT_EQ(output[4], 250.0F);
    EXPECT_FLOAT_EQ(output[5], 360.0F);
}

TEST(ElementwiseMulKernel, TrailingBroadcast) {
    constexpr float lhs[2] = {10.0F, 20.0F};
    constexpr float rhs[3] = {1.0F, 2.0F, 3.0F};
    float output[6] = {};
    constexpr int64_t lhs_shape[2] = {2, 1};
    constexpr int64_t lhs_strides[2] = {1, 1};
    constexpr int64_t rhs_shape[1] = {3};
    constexpr int64_t rhs_strides[1] = {1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 1};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs_tensor = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();

    EXPECT_FLOAT_EQ(output[0], 10.0F);
    EXPECT_FLOAT_EQ(output[1], 20.0F);
    EXPECT_FLOAT_EQ(output[2], 30.0F);
    EXPECT_FLOAT_EQ(output[3], 20.0F);
    EXPECT_FLOAT_EQ(output[4], 40.0F);
    EXPECT_FLOAT_EQ(output[5], 60.0F);
}

TEST(ElementwiseMulKernel, RankZeroScalars) {
    constexpr float lhs[1] = {3.0F};
    constexpr float rhs[1] = {5.0F};
    float output[1] = {};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{lhs, DataType::Float32(), IntArrayView{}, IntArrayView{}},
            .rhs_tensor = TensorView{rhs, DataType::Float32(), IntArrayView{}, IntArrayView{}},
            .output_tensor = MutableTensorView{output, DataType::Float32(), IntArrayView{}, IntArrayView{}},
    });
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_FLOAT_EQ(output[0], 15.0F);
}

TEST(ElementwiseMulKernel, ZeroElementOutputReturnsOk) {
    constexpr int64_t shape[2] = {2, 0};
    constexpr int64_t strides[2] = {0, 1};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{nullptr, DataType::Float32(), shape, strides},
            .rhs_tensor = TensorView{nullptr, DataType::Float32(), shape, strides},
            .output_tensor = MutableTensorView{nullptr, DataType::Float32(), shape, strides},
    });
    EXPECT_TRUE(status.ok()) << status.ToString();
}

TEST(ElementwiseMulKernel, RejectsIncompatibleBroadcastAtBindingTime) {
    constexpr float lhs[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    constexpr float rhs[12] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F,
                               7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    float output[6] = {};
    constexpr int64_t lhs_shape[2] = {2, 3};
    constexpr int64_t lhs_strides[2] = {3, 1};
    constexpr int64_t rhs_shape[2] = {4, 3};
    constexpr int64_t rhs_strides[2] = {3, 1};
    constexpr int64_t output_shape[2] = {2, 3};
    constexpr int64_t output_strides[2] = {3, 1};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{lhs, DataType::Float32(), lhs_shape, lhs_strides},
            .rhs_tensor = TensorView{rhs, DataType::Float32(), rhs_shape, rhs_strides},
            .output_tensor = MutableTensorView{output, DataType::Float32(), output_shape, output_strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

TEST(ElementwiseMulKernel, RejectsNonFloat32AtBindingTime) {
    constexpr double lhs[1] = {1.0};
    constexpr double rhs[1] = {2.0};
    double output[1] = {};
    constexpr int64_t shape[1] = {1};
    constexpr int64_t strides[1] = {1};

    const Status status = RunElementwiseMul(ElementwiseMulTestViews{
            .lhs_tensor = TensorView{lhs, DataType::Double(), shape, strides},
            .rhs_tensor = TensorView{rhs, DataType::Double(), shape, strides},
            .output_tensor = MutableTensorView{output, DataType::Double(), shape, strides},
    });
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument) << status.ToString();
}

} // namespace