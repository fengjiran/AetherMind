#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/operators/silu_op.h"

#include <gtest/gtest.h>

#include <variant>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// --- Validation / CheckInputSpecs ---

TEST(SiluOp, ValidateParamsReturnsOk) {
    const SiluOp op{SiluOp::Params{}};
    EXPECT_TRUE(op.ValidateParams().ok());
}

TEST(SiluOp, RejectsWrongArity) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = op.CheckInputSpecs(std::span(inputs));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, RejectsNonFloat32Input) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
    };
    const Status status = op.CheckInputSpecs(inputs);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, AcceptsArbitraryShape) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8, 16})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluOp, AcceptsRankZeroInput) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluOp, AcceptsZeroDimension) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({0, 3})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

// --- InferOutputShapes ---

TEST(SiluOp, InferOutputShapesRejectsWrongArity) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, InferOutputShapesRejectsNonFloat32) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, InfersIdenticalOutputShape) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(SiluOp, InfersRankZeroOutput) {
    const SiluOp op{SiluOp::Params{}};
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(SiluOp, InfersSymbolicOutputShape) {
    const SiluOp op{SiluOp::Params{}};
    const ShapeSymbol dim0 = ShapeSymbol::Create();
    const ShapeSymbol dim1 = ShapeSymbol::Create();
    const TensorSpec inputs[1] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{dim0, dim1})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    // Symbolic dimensions are preserved as-is (element-wise op).
    EXPECT_EQ(inference->outputs[0].shape[0], dim0);
    EXPECT_EQ(inference->outputs[0].shape[1], dim1);
}

// --- Prepare ---

namespace {

struct StubKernelState {
    bool called = false;
};

StubKernelState g_stub_state{};

Status StubSiluKernel(const KernelContext& /*ctx*/) noexcept {
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
            .op_type = OpType::kSilu,
            .fn = &StubSiluKernel,
            .attrs = {},
            .debug_name = "test::stub_silu",
    };
}

}// namespace

TEST(SiluOp, PrepareFailsWithNullBackend) {
    SiluOp op{SiluOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, PrepareSucceedsWithFakeBackend) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluOp op{SiluOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubSiluKernel);
}

TEST(SiluOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    SiluOp op{SiluOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(SiluOp, PrepareFailsWithNullKernelFn) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kSilu,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    SiluOp op{SiluOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

// --- Run ---

TEST(SiluOp, RunFailsBeforePrepare) {
    const SiluOp op{SiluOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(SiluOp, RunFailsWithWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluOp op{SiluOp::Params{}};
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

TEST(SiluOp, RunFailsWithWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluOp op{SiluOp::Params{}};
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
            MutableTensorView(dummy, DataType::Float32(), shape, strides),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(SiluOp, RunReturnsUnimplementedAndDoesNotInvokeKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    SiluOp op{SiluOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float in_data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float out_data[4] = {};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView(in_data, DataType::Float32(), shape, strides),
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

TEST(SiluOp, CreateFromRegistry) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kSilu,
            OpParams{SiluOp::Params{}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    ASSERT_NE(created.value(), nullptr);
    EXPECT_EQ(created.value()->Type(), OpType::kSilu);
    EXPECT_STREQ(created.value()->Name(), "Silu");
}

TEST(SiluOp, CreateFromRegistryWithWrongParams) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kSilu,
            OpParams{RmsNormParams{}});
    EXPECT_FALSE(created.ok());
    EXPECT_EQ(created.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluOp, CreateDefaultParamsFromRegistry) {
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kSilu);
    ASSERT_TRUE(params.ok()) << params.status().ToString();
    EXPECT_TRUE(std::holds_alternative<SiluOp::Params>(params.value()));
}

}// namespace
}// namespace aethermind
