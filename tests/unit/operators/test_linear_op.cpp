#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/linear_op.h"
#include "aethermind/operators/operator_context.h"

#include <gtest/gtest.h>

#include "backend/cpu/kernels/linear/linear_internal.h"
#include <array>
#include <cstddef>
#include <cstring>
#include <new>
#include <variant>
#include <vector>

namespace {
using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

// ===== ValidateParams / CheckInputSpecs / InferOutputShapes tests =====

TEST(LinearOp, ValidatesStaticInputContract) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(LinearOp, AcceptsRank1Input) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
    };

    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(LinearOp, RejectsRankZeroInput) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, RejectsRankZeroWeight) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, RejectsRank3Input) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({2, 4, 3})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 8})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, RejectsWeightRank1) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({8})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, RejectsStaticKMismatch) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 8})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({16, 16})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, InfersOutputShapeFromWeight) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4, 4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({11008, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 4);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 11008);
}

TEST(LinearOp, InfersRank1OutputShape) {
    const LinearOp op{LinearOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({4096})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({11008, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 1U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 11008);
}

TEST(LinearOp, EmitsRuntimeCheckForDistinctSymbolicK) {
    const LinearOp op{LinearOp::Params{}};
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol input_k = ShapeSymbol::Create();
    const ShapeSymbol out_features = ShapeSymbol::Create();
    const ShapeSymbol weight_k = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, input_k})},
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{out_features, weight_k})},
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
    EXPECT_EQ(equal.rhs.dim_index, 1U);
}

TEST(LinearOp, AcceptsSharedSymbolicK) {
    const LinearOp op{LinearOp::Params{}};
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol k = ShapeSymbol::Create();
    const ShapeSymbol out_features = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, k})},
            TensorSpec{.dtype = DataType::Float32(),
                       .shape = SymbolicShape(std::vector<ShapeSymbol>{out_features, k})},
    };

    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

// ===== Prepare/Run tests =====

struct StubKernelState {
    bool called = false;
    const void* kernel_params = nullptr;
    std::span<const std::byte> attrs{};
};

StubKernelState g_stub_state;

Status StubLinearKernel(const KernelContext& ctx) noexcept {
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

Status BuildStubLinearParams(std::span<const TensorView> inputs,
                             std::span<const MutableTensorView> outputs,
                             void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Linear requires 2 inputs and 1 output");
    }
    ::new (params_buffer) cpu::detail::LinearParams{
            .input_tensor = inputs[0],
            .weight_tensor = inputs[1],
            .output_tensor = outputs[0],
    };
    return Status::Ok();
}

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kLinear,
            .fn = &StubLinearKernel,
            .attrs = {},
            .debug_name = "test::stub_linear",
            .params_builder = &BuildStubLinearParams,
            .params_size = sizeof(cpu::detail::LinearParams),
    };
}

// RAII helper: owns dummy data and builds valid Linear StepTensorBinding.
// Shapes: input [2,4], weight [3,4], output [2,3].
// Must outlive any TensorView/MutableTensorView it produces.
struct LinearBindingBuilder {
    float data[12]{};
    std::array<int64_t, 2> input_shape{2, 4};
    std::array<int64_t, 2> input_strides{4, 1};
    std::array<int64_t, 2> weight_shape{3, 4};
    std::array<int64_t, 2> weight_strides{4, 1};
    std::array<int64_t, 2> output_shape{2, 3};
    std::array<int64_t, 2> output_strides{3, 1};

    StepTensorBinding Build() {
        StepTensorBinding b;
        b.inputs = {
                TensorView(data, DataType::Float32(), input_shape, input_strides),
                TensorView(data, DataType::Float32(), weight_shape, weight_strides),
        };
        b.outputs = {
                MutableTensorView(data, DataType::Float32(), output_shape, output_strides),
        };
        return b;
    }
};

TEST(LinearOp, PrepareResolvesKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    LinearOp op{LinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubLinearKernel);
    // LinearParams is empty; Prepare does not write attrs.
    EXPECT_TRUE(resolved.attrs.empty());
}

TEST(LinearOp, PrepareFailsWithNullBackend) {
    LinearOp op{LinearOp::Params{}};
    OperatorContext ctx{.backend = nullptr};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(LinearOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    LinearOp op{LinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(LinearOp, PrepareFailsWithNullKernelFn) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kLinear,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    LinearOp op{LinearOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

TEST(LinearOp, RunFailsBeforePrepare) {
    LinearOp op{LinearOp::Params{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;

    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(LinearOp, RunFailsWithWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    LinearOp op{LinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[4]{};
    std::array<int64_t, 2> shape_2d{2, 2};
    std::array<int64_t, 2> strides_2d{2, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), shape_2d, strides_2d),
            // Only 1 input; Linear requires 2.
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

TEST(LinearOp, RunFailsWithWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    LinearOp op{LinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    float dummy[12]{};
    std::array<int64_t, 2> input_shape{2, 4};
    std::array<int64_t, 2> input_strides{4, 1};
    std::array<int64_t, 2> weight_shape{3, 4};
    std::array<int64_t, 2> weight_strides{4, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy, DataType::Float32(), input_shape, input_strides),
            TensorView(dummy, DataType::Float32(), weight_shape, weight_strides),
    };
    step.outputs = {};// No outputs; Linear requires 1.
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(LinearOp, RunInvokesKernelAndReturnsOk) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    LinearOp op{LinearOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    LinearBindingBuilder builder;
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    // Executor sets attrs from ResolvedKernel before calling Run.
    kernel_ctx.attrs = op.GetResolvedKernel().attrs;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(g_stub_state.called);
    EXPECT_NE(g_stub_state.kernel_params, nullptr);
    // LinearParams is empty; attrs should be empty.
    EXPECT_TRUE(g_stub_state.attrs.empty());
}

}// namespace
