#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/matmul_op.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

#include <variant>

namespace {
using namespace aethermind;

// --- Prepare ---

struct StubKernelState {
    bool called = false;
};

StubKernelState g_stub_state{};

Status StubMatMulKernel(const KernelContext& /*ctx*/) noexcept {
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
            .op_type = OpType::kMatMul,
            .fn = &StubMatMulKernel,
            .attrs = {},
            .debug_name = "test::stub_matmul",
    };
}

TEST(MatMulOpPrepare, RejectsNullBackend) {
    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOpPrepare, ResolvesKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubMatMulKernel);
}

TEST(MatMulOpPrepare, PropagatesKernelResolutionFailure) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(MatMulOpPrepare, RejectsResolvedNullKernelFunction) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kMatMul,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

// --- Run ---

TEST(MatMulOpRun, RejectsCallBeforePrepare) {
    const MatMulOp op{MatMulOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(MatMulOpRun, RejectsWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    MatMulOp op{MatMulOp::Params{}};
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

TEST(MatMulOpRun, RejectsWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    MatMulOp op{MatMulOp::Params{}};
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

TEST(MatMulOpRun, ReturnsUnimplemented) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    MatMulOp op{MatMulOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float lhs_data[6] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
    float rhs_data[12] = {1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F, 7.0F, 8.0F, 9.0F, 10.0F, 11.0F, 12.0F};
    float out_data[8] = {};
    const int64_t lhs_shape[2] = {2, 3};
    const int64_t rhs_shape[2] = {3, 4};
    const int64_t out_shape[2] = {2, 4};
    const int64_t lhs_strides[2] = {3, 1};
    const int64_t rhs_strides[2] = {4, 1};
    const int64_t out_strides[2] = {4, 1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView(lhs_data, DataType::Float32(), lhs_shape, lhs_strides),
                                                     TensorView(rhs_data, DataType::Float32(), rhs_shape, rhs_strides),
                                             },
                                             .outputs = {
                                                     MutableTensorView(out_data, DataType::Float32(), out_shape, out_strides),
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

TEST(MatMulOp, CreateFromRegistry) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kMatMul,
            OpParams{MatMulOp::Params{}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    ASSERT_NE(created.value(), nullptr);
    EXPECT_EQ(created.value()->Type(), OpType::kMatMul);
    EXPECT_STREQ(created.value()->Name(), "MatMul");
}

TEST(MatMulOp, CreateFromRegistryWithTransposeRhs) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kMatMul,
            OpParams{MatMulOp::Params{.transpose_rhs = true}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    ASSERT_NE(created.value(), nullptr);
    EXPECT_EQ(created.value()->Type(), OpType::kMatMul);
}

TEST(MatMulOp, CreateFromRegistryWithWrongParams) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kMatMul,
            OpParams{RmsNormParams{}});
    EXPECT_FALSE(created.ok());
    EXPECT_EQ(created.status().code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, CreateDefaultParamsFromRegistry) {
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kMatMul);
    ASSERT_TRUE(params.ok()) << params.status().ToString();
    const auto* typed_params = std::get_if<MatMulOp::Params>(&params.value());
    ASSERT_NE(typed_params, nullptr);
    EXPECT_FALSE(typed_params->transpose_rhs);
}

}// namespace
