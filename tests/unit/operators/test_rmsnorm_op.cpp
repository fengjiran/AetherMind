#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/rmsnorm_op.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <variant>
#include <new>
#include "backend/cpu/kernels/rmsnorm/rmsnorm_internal.h"

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(RmsNormOp, ValidatesStaticInputContract) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(RmsNormOp, PreservesInputShapeAsOutputShape) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
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

TEST(RmsNormOp, EmitsRuntimeCheckForDistinctSymbolicHiddenDimension) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol input_hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, input_hidden})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{weight_hidden})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    ASSERT_EQ(inference->runtime_checks.size(), 1U);
    const ShapeConstraint& constraint = inference->runtime_checks[0];
    ASSERT_TRUE(std::holds_alternative<DimEqualConstraint>(constraint.condition));
    const auto& equal = std::get<DimEqualConstraint>(constraint.condition);
    EXPECT_EQ(equal.lhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.lhs.tensor_port.tensor_idx, 0U);
    EXPECT_EQ(equal.lhs.dim_index, 1U);
    EXPECT_EQ(equal.rhs.tensor_port.direction, TensorPortType::kInput);
    EXPECT_EQ(equal.rhs.tensor_port.tensor_idx, 1U);
    EXPECT_EQ(equal.rhs.dim_index, 0U);
}

TEST(RmsNormOp, AcceptsSharedSymbolicHiddenDimension) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const ShapeSymbol seq_len = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{seq_len, hidden_size})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{hidden_size})},
    };

    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(RmsNormOp, RejectsStaticHiddenMismatch) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormOp, RejectsRankZeroInput) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormOp, RejectsRankZeroWeight) {
    const RmsNormOp op{RmsNormOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

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
    ::new (params_buffer) cpu::CpuRmsNormParams{
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
            .params_size = sizeof(cpu::CpuRmsNormParams),
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

TEST(RmsNormOp, PrepareResolvesKernelAndWritesEps) {
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

TEST(RmsNormOp, PrepareFailsWithNullBackend) {
    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext ctx{.backend = nullptr};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(RmsNormOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    RmsNormOp op{RmsNormOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(RmsNormOp, PrepareFailsWithNullKernelFn) {
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

TEST(RmsNormOp, RunFailsBeforePrepare) {
    RmsNormOp op{RmsNormOp::Params{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;

    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(RmsNormOp, RunFailsWithWrongInputCount) {
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

TEST(RmsNormOp, RunFailsWithWrongOutputCount) {
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

TEST(RmsNormOp, RunInvokesKernelAndReturnsOk) {
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
}// namespace aethermind
