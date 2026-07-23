#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/elementwise_mul_op.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "backend/cpu/kernels/elementwise_mul/elementwise_mul_internal.h"

#include <cstring>
#include <gtest/gtest.h>
#include <new>
#include <variant>

namespace {
using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// --- Validation ---

TEST(ElementwiseMulOp, InferOperatorAcceptsValidParams) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

TEST(ElementwiseMulOp, RejectsWrongArity) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, RejectsNonFloat32Input) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, RejectsStaticIncompatibleBroadcast) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3})},
    };
    const Status status = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, RejectsUnrankedInput) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    const Status status = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, AcceptsSameShape) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

TEST(ElementwiseMulOp, AcceptsBroadcastShape) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 1})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

TEST(ElementwiseMulOp, AcceptsDifferentRankBroadcast) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

TEST(ElementwiseMulOp, AcceptsRankZeroInput) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

TEST(ElementwiseMulOp, AcceptsZeroDimension) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({0, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({0, 3})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs).status().ok());
}

// --- Inference ---

TEST(ElementwiseMulOp, InferOperatorRejectsNonFloat32) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, InfersSameShapeOutput) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 8);
}

TEST(ElementwiseMulOp, InfersBroadcastOutputShape) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 1})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(ElementwiseMulOp, InfersRankZeroOutput) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_TRUE(inference->outputs[0].shape.IsRankZero());
}

TEST(ElementwiseMulOp, InfersDifferentRankBroadcast) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok());
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
}

TEST(ElementwiseMulOp, EmitsDeferredDimBroadcastableConstraints) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    const ShapeSymbol lhs_dim0 = ShapeSymbol::Create();
    const ShapeSymbol lhs_dim1 = ShapeSymbol::Create();
    const ShapeSymbol rhs_dim0 = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{lhs_dim0, lhs_dim1})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_dim0})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{ElementwiseMulOp::Params{}}, inputs);

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
    const void* kernel_params = nullptr;
    bool lhs_valid = false;
    bool rhs_valid = false;
    bool output_valid = false;
};

StubKernelState g_stub_state{};

Status StubElementwiseMulKernel(const KernelContext& ctx) noexcept {
    g_stub_state.called = true;
    g_stub_state.kernel_params = ctx.kernel_params;
    // `ctx.kernel_params` points at the cpu::detail::ElementwiseMulParams owned by
    // ElementwiseMulOp::Run's stack frame, valid only for the duration of
    // this call. Validate here rather than after Run() returns, where the
    // pointer would dangle.
    const auto* params = static_cast<const cpu::detail::ElementwiseMulParams*>(ctx.kernel_params);
    g_stub_state.lhs_valid = params->lhs_tensor.is_valid();
    g_stub_state.rhs_valid = params->rhs_tensor.is_valid();
    g_stub_state.output_valid = params->output_tensor.is_valid();
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

Status BuildStubElementwiseMulParams(std::span<const TensorView> inputs,
                                     std::span<const MutableTensorView> outputs,
                                     void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("ElementwiseMul requires 2 inputs and 1 output");
    }
    ::new (params_buffer) cpu::detail::ElementwiseMulParams{
            .lhs_tensor = inputs[0],
            .rhs_tensor = inputs[1],
            .output_tensor = outputs[0],
    };
    return Status::Ok();
}

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kElementwiseMul,
            .fn = &StubElementwiseMulKernel,
            .attrs = {},
            .debug_name = "test::stub_elementwise_mul",
            .params_builder = &BuildStubElementwiseMulParams,
            .params_size = sizeof(cpu::detail::ElementwiseMulParams),
    };
}

}// namespace

TEST(ElementwiseMulOp, PrepareFailsWithNullBackend) {
    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, PrepareSucceedsWithFakeBackend) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubElementwiseMulKernel);
}

TEST(ElementwiseMulOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(ElementwiseMulOp, PrepareFailsWithNullKernelFn) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kElementwiseMul,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

// --- Run ---

TEST(ElementwiseMulOp, RunFailsBeforePrepare) {
    const ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(ElementwiseMulOp, RunFailsWithWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
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

TEST(ElementwiseMulOp, RunFailsWithWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
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

TEST(ElementwiseMulOp, RunInvokesKernelAndReturnsOk) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    ElementwiseMulOp op{ElementwiseMulOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float lhs_data[4] = {1.0F, 2.0F, 3.0F, 4.0F};
    float rhs_data[4] = {10.0F, 20.0F, 30.0F, 40.0F};
    float out_data[4] = {};
    const int64_t shape[2] = {2, 2};
    const int64_t strides[2] = {2, 1};

    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {
                                                     TensorView(lhs_data, DataType::Float32(), shape, strides),
                                                     TensorView(rhs_data, DataType::Float32(), shape, strides),
                                             },
                                             .outputs = {
                                                     MutableTensorView(out_data, DataType::Float32(), shape, strides),
                                             },
                                     });

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(g_stub_state.called);
    EXPECT_NE(g_stub_state.kernel_params, nullptr);

    const auto* params = static_cast<const cpu::detail::ElementwiseMulParams*>(g_stub_state.kernel_params);
    EXPECT_TRUE(g_stub_state.lhs_valid);
    EXPECT_TRUE(g_stub_state.rhs_valid);
    EXPECT_TRUE(g_stub_state.output_valid);
    (void) params;// captured only to confirm non-null above
}

// --- Registry ---

TEST(ElementwiseMulOp, CreateFromRegistry) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kElementwiseMul,
            OpParams{ElementwiseMulOp::Params{}});
    ASSERT_TRUE(created.ok()) << created.status().ToString();
    ASSERT_NE(created.value(), nullptr);
    EXPECT_EQ(created.value()->Type(), OpType::kElementwiseMul);
    EXPECT_STREQ(created.value()->Name(), "ElementwiseMul");
}

TEST(ElementwiseMulOp, CreateFromRegistryWithWrongParams) {
    const StatusOr<std::unique_ptr<Operator>> created = OperatorRegistry::Create(
            OpType::kElementwiseMul,
            OpParams{RmsNormParams{}});
    EXPECT_FALSE(created.ok());
    EXPECT_EQ(created.status().code(), StatusCode::kInvalidArgument);
}

TEST(ElementwiseMulOp, CreateDefaultParamsFromRegistry) {
    const StatusOr<OpParams> params = OperatorRegistry::CreateDefaultParams(OpType::kElementwiseMul);
    ASSERT_TRUE(params.ok()) << params.status().ToString();
    EXPECT_TRUE(std::holds_alternative<ElementwiseMulOp::Params>(params.value()));
}

}// namespace
