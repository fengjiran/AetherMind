#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/silu_mul_op.h"

#include <gtest/gtest.h>

#include <variant>

namespace {
using namespace aethermind;

// --- Prepare ---

struct StubKernelState {
    bool called = false;
};

StubKernelState g_stub_state{};

Status StubSiluMulKernel(const KernelContext& /*ctx*/) noexcept {
    g_stub_state.called = true;
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

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kSiluMul,
            .fn = &StubSiluMulKernel,
            .attrs = {},
            .debug_name = "test::stub_silu_mul",
    };
}

TEST(SiluMulOpPrepare, RejectsNullBackend) {
    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOpPrepare, ResolvesKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubSiluMulKernel);
}

TEST(SiluMulOpPrepare, PropagatesKernelResolutionFailure) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(SiluMulOpPrepare, RejectsResolvedNullKernelFunction) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kSiluMul,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

// --- Run ---

TEST(SiluMulOpRun, RejectsCallBeforePrepare) {
    const SiluMulOp op{SiluMulOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(SiluMulOpRun, RejectsWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[4]{};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape, strides),
    };
    step.outputs = {
            MutableTensorView(dummy, DataType::Float32(), shape, strides),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(SiluMulOpRun, RejectsWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[4]{};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape, strides),
            TensorView(dummy, DataType::Float32(), shape, strides),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(SiluMulOpRun, ReturnsUnimplemented) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float gate_data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float up_data[4] = {10.0F, 20.0F, 30.0F, 40.0F};
    float out_data[4] = {};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView(gate_data, DataType::Float32(), shape, strides),
                                                     TensorView(up_data, DataType::Float32(), shape, strides),
                                             },
                                             .outputs = {
                                                     MutableTensorView(out_data, DataType::Float32(), shape, strides),
                                             },
                                     });

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kUnimplemented);
    // Stub kernel must not be invoked: Run returns Unimplemented after binding
    // validation without dispatching to the kernel.
    EXPECT_FALSE(g_stub_state.called);
}

// --- Registry ---

TEST(SiluMulOp, CreateFromRegistry) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kSiluMul,
            OpParams{SiluMulOp::Params{}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    ASSERT_NE(created.value(), nullptr);
    EXPECT_EQ(created.value()->Type(), OpType::kSiluMul);
    EXPECT_STREQ(created.value()->Name(), "SiluMul");
}

TEST(SiluMulOp, CreateFromRegistryWithWrongParams) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kSiluMul,
            OpParams{RmsNormParams{}});
    EXPECT_FALSE(created.ok());
    EXPECT_EQ(created.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, CreateDefaultParamsFromRegistry) {
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kSiluMul);
    ASSERT_TRUE(params.ok()) << params.status().ToString();
    EXPECT_TRUE(std::holds_alternative<SiluMulOp::Params>(params.value()));
}

}// namespace
