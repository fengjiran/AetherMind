#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/qkv_linear_op.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

// ===== Prepare/Run tests =====

struct StubKernelState {
    bool called = false;
    const void* kernel_params = nullptr;
    std::span<const std::byte> attrs{};
};

StubKernelState g_stub_state;

// Test-only params object populated by BuildStubQkvLinearParams; the QkvLinear
// operator itself is kernel-agnostic and only forwards views to the resolver.
struct StubQkvLinearParams {
    int dummy = 0;
};

Status StubQkvLinearKernel(const KernelContext& ctx) noexcept {
    g_stub_state.called = true;
    g_stub_state.kernel_params = ctx.kernel_params;
    g_stub_state.attrs = ctx.attrs;
    return Status::Ok();
}

void ResetStubState() {
    g_stub_state = StubKernelState{};
}

class FakeBackend final : public Backend {
public:
    StatusOr<ResolvedKernel> resolve_result{Status::NotFound("unconfigured")};

    AM_NODISCARD DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override {
        static const BackendCapabilities kCaps{};
        return kCaps;
    }
    AM_NODISCARD KernelFunc ResolveKernel(OpType, const KernelSelector&) const noexcept override {
        return resolve_result.ok() ? resolve_result.value().fn : nullptr;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType, const KernelSelector&) const noexcept override {
        return resolve_result;
    }
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

Status BuildStubQkvLinearParams(std::span<const TensorView> inputs,
                                std::span<const MutableTensorView> outputs,
                                void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 3) {
        return Status::InvalidArgument("QkvLinear requires 2 inputs and 3 outputs");
    }
    ::new (params_buffer) StubQkvLinearParams{};
    return Status::Ok();
}

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kQkvLinear,
            .fn = &StubQkvLinearKernel,
            .attrs = {},
            .debug_name = "test::stub_qkv_linear",
            .params_builder = &BuildStubQkvLinearParams,
            .params_size = sizeof(StubQkvLinearParams),
    };
}

// RAII helper: owns dummy data and builds a valid QkvLinear StepTensorBinding.
// Shapes: input [2,4], qkv_weight [32,4], outputs q [2,16], k [2,8], v [2,8].
// Must outlive any TensorView/MutableTensorView it produces.
struct QkvLinearBindingBuilder {
    float data[200]{};
    std::array<int64_t, 2> input_shape{2, 4};
    std::array<int64_t, 2> input_strides{4, 1};
    std::array<int64_t, 2> weight_shape{32, 4};
    std::array<int64_t, 2> weight_strides{4, 1};
    std::array<int64_t, 2> q_shape{2, 16};
    std::array<int64_t, 2> k_shape{2, 8};
    std::array<int64_t, 2> v_shape{2, 8};
    std::array<int64_t, 2> q_strides{16, 1};
    std::array<int64_t, 2> k_strides{8, 1};
    std::array<int64_t, 2> v_strides{8, 1};

    StepTensorBinding Build() {
        StepTensorBinding b;
        b.inputs = {
                TensorView(data, DataType::Float32(), input_shape, input_strides),
                TensorView(data, DataType::Float32(), weight_shape, weight_strides),
        };
        b.outputs = {
                MutableTensorView(data, DataType::Float32(), q_shape, q_strides),
                MutableTensorView(data, DataType::Float32(), k_shape, k_strides),
                MutableTensorView(data, DataType::Float32(), v_shape, v_strides),
        };
        return b;
    }
};

TEST(QkvLinearOpPrepare, ResolvesKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubQkvLinearKernel);
    EXPECT_TRUE(resolved.attrs.empty());
}

TEST(QkvLinearOpPrepare, RejectsNullBackend) {
    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext ctx{.backend = nullptr};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(QkvLinearOpPrepare, PropagatesKernelResolutionFailure) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(QkvLinearOpPrepare, RejectsResolvedNullKernelFunction) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kQkvLinear,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

TEST(QkvLinearOpRun, RejectsCallBeforePrepare) {
    QkvLinearOp op{QkvLinearOp::Params{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;

    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(QkvLinearOpRun, RejectsWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[200]{};
    std::array<int64_t, 2> shape_2d{2, 4};
    std::array<int64_t, 2> strides_2d{4, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            // Only 1 input; QkvLinear requires 2.
    };
    step.outputs = {
            MutableTensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            MutableTensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            MutableTensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(QkvLinearOpRun, RejectsWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[200]{};
    std::array<int64_t, 2> input_shape{2, 4};
    std::array<int64_t, 2> input_strides{4, 1};
    std::array<int64_t, 2> weight_shape{32, 4};
    std::array<int64_t, 2> weight_strides{4, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), input_shape, input_strides),
            TensorView(dummy, DataType::Float32(), weight_shape, weight_strides),
    };
    step.outputs = {};// No outputs; QkvLinear requires 3.
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(QkvLinearOpRun, InvokesResolvedKernelAndForwardsAttributes) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    QkvLinearOp op{QkvLinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    QkvLinearBindingBuilder builder;
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    // Executor sets attrs from ResolvedKernel before calling Run.
    kernel_ctx.attrs = op.GetResolvedKernel().attrs;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(g_stub_state.called);
    EXPECT_NE(g_stub_state.kernel_params, nullptr);
    EXPECT_TRUE(g_stub_state.attrs.empty());
}

}// namespace
