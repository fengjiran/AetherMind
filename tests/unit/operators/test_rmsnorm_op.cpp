#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/rmsnorm_op.h"
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

#include <cstring>
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

Status StubRmsNormKernel(const KernelContext& ctx) noexcept {
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
    AM_NODISCARD StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType, const KernelSelector&) const noexcept override {
        return resolve_result;
    }
    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }
};

Status BuildStubRmsNormParams(std::span<const TensorView> inputs,
                              std::span<const MutableTensorView> outputs,
                              void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("RmsNorm requires 2 inputs and 1 output");
    }
    ::new (params_buffer) cpu::detail::RmsNormParams{
            .input_tensor = inputs[0],
            .weight_tensor = inputs[1],
            .output_tensor = outputs[0],
    };
    return Status::Ok();
}

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kRmsNorm,
            .fn = &StubRmsNormKernel,
            .attrs = {},
            .debug_name = "test::stub_rmsnorm",
            .params_builder = &BuildStubRmsNormParams,
            .params_size = sizeof(cpu::detail::RmsNormParams),
    };
}

// RAII helper: owns dummy data and builds valid RmsNorm StepTensorBinding.
// Must outlive any TensorView/MutableTensorView it produces.
struct RmsNormBindingBuilder {
    float data[8]{};
    std::array<int64_t, 2> shape_2d{4, 2};
    std::array<int64_t, 2> strides_2d{2, 1};
    std::array<int64_t, 1> shape_1d{2};
    std::array<int64_t, 1> strides_1d{1};

    StepTensorBinding Build() {
        StepTensorBinding b;
        b.inputs = {
                TensorView(data, DataType::Float32(), shape_2d, strides_2d),
                TensorView(data, DataType::Float32(), shape_1d, strides_1d),
        };
        b.outputs = {
                MutableTensorView(data, DataType::Float32(), shape_2d, strides_2d),
        };
        return b;
    }
};

TEST(RmsNormOpPrepare, ResolvesKernelAndWritesEpsilon) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    RmsNormOp op{RmsNormParams{.eps = 1.0e-5f}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubRmsNormKernel);
    // Prepare overwrites attrs with the 4-byte eps float.
    ASSERT_EQ(resolved.attrs.size(), sizeof(float));
    float eps = 0.0f;
    std::memcpy(&eps, resolved.attrs.data(), sizeof(float));
    EXPECT_FLOAT_EQ(eps, 1.0e-5f);
}

TEST(RmsNormOpPrepare, RejectsNullBackend) {
    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext ctx{.backend = nullptr};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormOpPrepare, PropagatesKernelResolutionFailure) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(RmsNormOpPrepare, RejectsResolvedNullKernelFunction) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kRmsNorm,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

TEST(RmsNormOpRun, RejectsCallBeforePrepare) {
    RmsNormOp op{RmsNormOp::Params{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;

    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(RmsNormOpRun, RejectsWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[4]{};
    std::array<int64_t, 2> shape_2d{2, 2};
    std::array<int64_t, 2> strides_2d{2, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            // Only 1 input; RmsNorm requires 2.
    };
    step.outputs = {
            MutableTensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(RmsNormOpRun, RejectsWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[4]{};
    std::array<int64_t, 2> shape_2d{2, 2};
    std::array<int64_t, 2> strides_2d{2, 1};
    std::array<int64_t, 1> shape_1d{2};
    std::array<int64_t, 1> strides_1d{1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            TensorView(dummy, DataType::Float32(), shape_1d, strides_1d),
    };
    step.outputs = {};// No outputs; RmsNorm requires 1.
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(RmsNormOpRun, InvokesResolvedKernelAndForwardsAttributes) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    RmsNormOp op{RmsNormParams{.eps = 1.0e-5f}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    RmsNormBindingBuilder builder;
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    // Executor sets attrs from ResolvedKernel before calling Run.
    kernel_ctx.attrs = op.GetResolvedKernel().attrs;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(g_stub_state.called);
    EXPECT_NE(g_stub_state.kernel_params, nullptr);
    // attrs should carry the 4-byte eps written by Prepare.
    ASSERT_EQ(g_stub_state.attrs.size(), sizeof(float));
}

}// namespace
