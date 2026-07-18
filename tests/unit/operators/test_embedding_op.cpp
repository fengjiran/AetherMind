#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <new>
#include "aethermind/backend/cpu/kernels/cpu_embedding_kernel.h"

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(EmbeddingOp, ValidatesInputContract) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({2})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeFromTokenIdsAndWeight) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({5})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, RejectsRankZeroTokenIds) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingOp, RejectsRankZeroWeight) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = StaticShape({2})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };

    const Status status = op.CheckInputSpecs(inputs);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingOp, AcceptsUint32TokenIds) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = StaticShape({2})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(op.ValidateParams().ok());
    EXPECT_TRUE(op.CheckInputSpecs(inputs).ok());
}

TEST(EmbeddingOp, InfersOutputShapeWithUint32Tokens) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::UInt(32), .shape = StaticShape({5})},
            TensorSpec{.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, PreservesSymbolicTokenAndHiddenDims) {
    const EmbeddingOp op{EmbeddingOp::Params{}};
    const ShapeSymbol token_count = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            TensorSpec{.dtype = DataType::Int(64), .shape = SymbolicShape(std::vector<ShapeSymbol>{token_count})},
            TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden_size})},
    };

    const StatusOr<InferenceResult> inference = op.InferOutputShapes(inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0], token_count);
    EXPECT_EQ(inference->outputs[0].shape[1], hidden_size);
}

TEST(EmbeddingOp, RegistryCreatesDefaultEmbeddingOperator) {
    StatusOr<std::unique_ptr<Operator>> op = OperatorRegistry::Create(
            OpType::kEmbedding,
            EmbeddingOp::Params{});

    ASSERT_TRUE(op.ok()) << op.status().ToString();
    ASSERT_NE(op.value(), nullptr);
    EXPECT_EQ(op.value()->Type(), OpType::kEmbedding);
    EXPECT_STREQ(op.value()->Name(), "Embedding");
}

// ===== Prepare/Run tests =====

struct StubKernelState {
    bool called = false;
    const void* kernel_params = nullptr;
};

StubKernelState g_stub_state;

Status StubEmbeddingKernel(const KernelContext& ctx) noexcept {
    g_stub_state.called = true;
    g_stub_state.kernel_params = ctx.kernel_params;
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

Status BuildStubEmbeddingParams(std::span<const TensorView> inputs,
                                    std::span<const MutableTensorView> outputs,
                                    void* params_buffer) noexcept {
    if (inputs.size() != 2 || outputs.size() != 1) {
        return Status::InvalidArgument("Embedding requires 2 inputs and 1 output");
    }
    ::new (params_buffer) CpuEmbeddingParams{
            .token_ids_ = inputs[0],
            .weight_ = inputs[1],
            .output_ = outputs[0],
    };
    return Status::Ok();
}

ResolvedKernel MakeStubKernel() {
    return ResolvedKernel{
            .op_type = OpType::kEmbedding,
            .fn = &StubEmbeddingKernel,
            .attrs = {},
            .debug_name = "test::stub_embedding",
            .params_builder = &BuildStubEmbeddingParams,
            .params_size = sizeof(CpuEmbeddingParams),
    };
}

// RAII helper: owns dummy data and builds valid Embedding StepTensorBinding.
// Must outlive any TensorView/MutableTensorView it produces.
struct EmbeddingBindingBuilder {
    int64_t token_ids[4]{1, 2, 3, 4};
    float weight[8]{};
    float output[8]{};
    std::array<int64_t, 1> shape_1d{4};
    std::array<int64_t, 1> strides_1d{1};
    std::array<int64_t, 2> shape_2d{4, 2};
    std::array<int64_t, 2> strides_2d{2, 1};

    StepTensorBinding Build() {
        StepTensorBinding b;
        b.inputs = {
                TensorView(token_ids, DataType::Int(64), shape_1d, strides_1d),
                TensorView(weight, DataType::Float32(), shape_2d, strides_2d),
        };
        b.outputs = {
                MutableTensorView(output, DataType::Float32(), shape_2d, strides_2d),
        };
        return b;
    }
};

TEST(EmbeddingOp, PrepareResolvesKernel) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    ASSERT_TRUE(status.ok()) << status.ToString();
    const ResolvedKernel& resolved = op.GetResolvedKernel();
    EXPECT_NE(resolved.fn, nullptr);
    EXPECT_EQ(resolved.fn, &StubEmbeddingKernel);
}

TEST(EmbeddingOp, PrepareFailsWithNullBackend) {
    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext ctx{.backend = nullptr};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingOp, PrepareFailsWhenKernelResolveFails) {
    FakeBackend backend;
    backend.resolve_result = Status::NotFound("test: kernel not found");

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kNotFound);
}

TEST(EmbeddingOp, PrepareFailsWithNullKernelFn) {
    FakeBackend backend;
    backend.resolve_result = ResolvedKernel{
            .op_type = OpType::kEmbedding,
            .fn = nullptr,
            .attrs = {},
            .debug_name = "test::null_fn",
    };

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext ctx{.backend = &backend};

    const Status status = op.Prepare(ctx);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInternal);
}

TEST(EmbeddingOp, RunFailsBeforePrepare) {
    EmbeddingOp op{EmbeddingOp::Params{}};
    KernelContext kernel_ctx;
    RuntimeBindingContext bindings;

    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(EmbeddingOp, RunFailsWithWrongInputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    int64_t dummy_tokens[4]{};
    std::array<int64_t, 1> shape_1d{4};
    std::array<int64_t, 1> strides_1d{1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy_tokens, DataType::Int(64), shape_1d, strides_1d),
            // Only 1 input; Embedding requires 2.
    };
    step.outputs = {
            MutableTensorView(dummy_tokens, DataType::Float32(), shape_1d, strides_1d),
    };
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(EmbeddingOp, RunFailsWithWrongOutputCount) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    int64_t dummy_tokens[4]{};
    float dummy_weight[8]{};
    std::array<int64_t, 1> shape_1d{4};
    std::array<int64_t, 1> strides_1d{1};
    std::array<int64_t, 2> shape_2d{4, 2};
    std::array<int64_t, 2> strides_2d{2, 1};

    RuntimeBindingContext bindings;
    StepTensorBinding step;
    step.inputs = {
            TensorView(dummy_tokens, DataType::Int(64), shape_1d, strides_1d),
            TensorView(dummy_weight, DataType::Float32(), shape_2d, strides_2d),
    };
    step.outputs = {};// No outputs; Embedding requires 1.
    bindings.SetStepTensorBinding(0, std::move(step));

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_FALSE(g_stub_state.called);
}

TEST(EmbeddingOp, RunInvokesKernelAndReturnsOk) {
    ResetStubState();
    FakeBackend backend;
    backend.resolve_result = MakeStubKernel();

    EmbeddingOp op{EmbeddingOp::Params{}};
    OperatorContext op_ctx{.backend = &backend};
    ASSERT_TRUE(op.Prepare(op_ctx).ok());

    EmbeddingBindingBuilder builder;
    RuntimeBindingContext bindings;
    bindings.SetStepTensorBinding(0, builder.Build());

    KernelContext kernel_ctx;
    const Status status = op.Run(kernel_ctx, bindings, 0);

    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_TRUE(g_stub_state.called);
    EXPECT_NE(g_stub_state.kernel_params, nullptr);
}

}// namespace
}// namespace aethermind
