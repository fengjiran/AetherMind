#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/matmul_op.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

#include <variant>

namespace {
using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// --- Validation ---

TEST(MatMulOp, InferOperatorAcceptsValidParams) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status().ok());
}

TEST(MatMulOp, RejectsWrongArity) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[3] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, std::span(inputs)).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsNonFloat32Input) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsUnrankedLhs) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsUnrankedRhs) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape{}},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsRank1Lhs) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsRank1Rhs) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsStaticInnerMismatch) {
    const MatMulOp op{MatMulOp::Params{}};// transpose_rhs=false
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 5})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsStaticInnerMismatchWithTransposeRhs) {
    const MatMulOp op{MatMulOp::Params{.transpose_rhs = true}};// rhs layout [..., N, K]
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},// K=3
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 5})},// K=5 -> mismatch
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, RejectsStaticIncompatibleBatch) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3, 5})},
    };
    const Status status = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status();
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, AcceptsRank2Inputs) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status().ok());
}

TEST(MatMulOp, AcceptsBatchedMatMul) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({5, 2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({5, 3, 4})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status().ok());
}

TEST(MatMulOp, AcceptsBroadcastBatch) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3, 5})},
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs).status().ok());
}

TEST(MatMulOp, AcceptsTransposeRhs) {
    const MatMulOp op{MatMulOp::Params{.transpose_rhs = true}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3})},// [N, K] = [4, 3]
    };
    EXPECT_TRUE(InferOperator(op.Type(), OpParams{MatMulOp::Params{.transpose_rhs = true}}, inputs).status().ok());
}

// --- Inference ---

TEST(MatMulOp, InferOperatorRejectsNonFloat32) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Int(32), .shape = StaticShape({3, 4})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);
    EXPECT_FALSE(inference.ok());
    EXPECT_EQ(inference.status().code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, InfersRank2Output) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 4})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4);
}

TEST(MatMulOp, InfersRank2OutputWithTransposeRhs) {
    const MatMulOp op{MatMulOp::Params{.transpose_rhs = true}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3})},// [M, K]
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3})},// [N, K]
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{.transpose_rhs = true}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);// M
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4);// N
}

TEST(MatMulOp, InfersBatchedOutput) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({5, 2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({5, 3, 4})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 4);
}

TEST(MatMulOp, InfersBroadcastBatchOutput) {
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({1, 2, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 3, 5})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);// broadcast batch
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 5);
}

TEST(MatMulOp, InfersDifferentRankBatchBroadcast) {
    // lhs batch [2], rhs batch [] -> output batch [2]
    const MatMulOp op{MatMulOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 3, 4})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 5})},
    };
    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);
    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 3);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 5);
}

TEST(MatMulOp, EmitsInnerDimEqualConstraintForSymbolicInner) {
    const MatMulOp op{MatMulOp::Params{}};
    const ShapeSymbol lhs_inner = ShapeSymbol::Create();
    const ShapeSymbol rhs_inner = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), lhs_inner})},
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_inner, ShapeSymbol::CreateFromValue(4)})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& eq = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(eq.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(eq.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(eq.lhs.dim_index, 1U);// lhs_inner at axis 1
    EXPECT_EQ(eq.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(eq.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(eq.rhs.dim_index, 0U);// rhs_inner (K) at axis 0 (transpose_rhs=false)
}

TEST(MatMulOp, EmitsInnerDimEqualConstraintForTransposeRhs) {
    const MatMulOp op{MatMulOp::Params{.transpose_rhs = true}};
    const ShapeSymbol lhs_inner = ShapeSymbol::Create();
    const ShapeSymbol rhs_inner = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(2), lhs_inner})},
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::CreateFromValue(4), rhs_inner})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{.transpose_rhs = true}}, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& eq = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(eq.lhs.dim_index, 1U);
    // transpose_rhs=true: rhs layout [N, K], K at axis 1
    EXPECT_EQ(eq.rhs.dim_index, 1U);
}

TEST(MatMulOp, EmitsBatchBroadcastableConstraintForSymbolicBatch) {
    const MatMulOp op{MatMulOp::Params{}};
    const ShapeSymbol lhs_batch = ShapeSymbol::Create();
    const ShapeSymbol rhs_batch = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{lhs_batch, ShapeSymbol::CreateFromValue(2), ShapeSymbol::CreateFromValue(3)})},
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{rhs_batch, ShapeSymbol::CreateFromValue(3), ShapeSymbol::CreateFromValue(4)})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(op.Type(), OpParams{MatMulOp::Params{}}, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Inner dims are static 3 == 3, so only the batch dim emits a constraint.
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimBroadcastableConstraint>(constraint.condition));
    const auto& bc = std::get<DimBroadcastableConstraint>(constraint.condition);
    EXPECT_EQ(bc.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(bc.lhs.dim_index, 0U);// batch axis 0 on lhs
    EXPECT_EQ(bc.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(bc.rhs.dim_index, 0U);// batch axis 0 on rhs
}

// --- Prepare ---

namespace {

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

}// namespace

TEST(MatMulOp, PrepareFailsWithNullBackend) {
    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{};
    ctx.backend = nullptr;
    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(MatMulOp, PrepareSucceedsWithFakeBackend) {
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

TEST(MatMulOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    MatMulOp op{MatMulOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(MatMulOp, PrepareFailsWithNullKernelFn) {
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

TEST(MatMulOp, RunFailsBeforePrepare) {
    const MatMulOp op{MatMulOp::Params{}};
    KernelContext ctx{};
    RuntimeBindingContext bindings;
    const Status status = op.Run(ctx, bindings, 0);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(MatMulOp, RunFailsWithWrongInputCount) {
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

TEST(MatMulOp, RunFailsWithWrongOutputCount) {
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

TEST(MatMulOp, RunReturnsUnimplementedAndDoesNotInvokeKernel) {
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
