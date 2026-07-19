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

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// --- Validation / CheckInputSpecs ---

TEST(SiluMulOp, ValidateParamsReturnsOk) {
    const SiluMulOp op{SiluMulOp::Params{}};
    EXPECT_TRUE(op.ValidateParams().ok());
}

TEST(SiluMulOp, RejectsWrongArity) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = op.CheckInputSpecs(std::span(inputs));
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, RejectsNonFloat32Input) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = op.CheckInputSpecs(inputs);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, RejectsStaticIncompatibleBroadcast) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3})},
    };
    const Status status = op.CheckInputSpecs(inputs);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, RejectsUnrankedInput) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = op.CheckInputSpecs(inputs);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, AcceptsSameShape) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluMulOp, AcceptsBroadcastShape) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 1})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluMulOp, AcceptsDifferentRankBroadcast) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluMulOp, AcceptsRankZeroInput) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(SiluMulOp, AcceptsZeroDimension) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({0, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({0, 3})},
    };
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

// --- InferOutputShapes ---

TEST(SiluMulOp, InferOutputShapesRejectsNonFloat32) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, InfersSameShapeOutput) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
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

TEST(SiluMulOp, InfersBroadcastOutputShape) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 1})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(SiluMulOp, InfersRankZeroOutput) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(SiluMulOp, InfersDifferentRankBroadcast) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
    };
    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(SiluMulOp, EmitsDeferredDimBroadcastableConstraints) {
    const SiluMulOp op{SiluMulOp::Params{}};
    const ShapeSymbol gate_dim0 = ShapeSymbol::Create();
    const ShapeSymbol gate_dim1 = ShapeSymbol::Create();
    const ShapeSymbol up_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{gate_dim0, gate_dim1})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{up_dim0})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(constraint.condition));
    const auto& bc = std::get<DimBroadcastableConstraint>(constraint.condition);
    EXPECT_EQ(bc.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(bc.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(bc.lhs.dim_index, 1U);
    EXPECT_EQ(bc.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(bc.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(bc.rhs.dim_index, 0U);
}

// --- Prepare ---

namespace {

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

}// namespace

TEST(SiluMulOp, PrepareFailsWithNullBackend) {
    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(SiluMulOp, PrepareSucceedsWithFakeBackend) {
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

TEST(SiluMulOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    SiluMulOp op{SiluMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(SiluMulOp, PrepareFailsWithNullKernelFn) {
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

TEST(SiluMulOp, RunFailsBeforePrepare) {
    const SiluMulOp op{SiluMulOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(SiluMulOp, RunFailsWithWrongInputCount) {
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

TEST(SiluMulOp, RunFailsWithWrongOutputCount) {
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

TEST(SiluMulOp, RunReturnsUnimplementedAndDoesNotInvokeKernel) {
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

}  // namespace
