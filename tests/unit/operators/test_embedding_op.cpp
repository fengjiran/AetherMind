#include "aethermind/backend/backend.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/operators/embedding_op.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/operators/operator_registry.h"
#include "backend/cpu/kernels/embedding/embedding_internal.h"

#include <gtest/gtest.h>

namespace {
using namespace aethermind;

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

TEST(EmbeddingOp, ValidatesInputContract) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({2})},
            {.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingOp, InfersOutputShapeFromTokenIdsAndWeight) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({5})},
            {.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, RejectsRankZeroTokenIds) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
            {.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    const Status status = InferOperator(OpType::kEmbedding, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingOp, RejectsRankZeroWeight) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({2})},
            {.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{})},
    };

    const Status status = InferOperator(OpType::kEmbedding, params, inputs).status();

    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(EmbeddingOp, AcceptsUint32TokenIds) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::UInt(32), .shape = StaticShape({2})},
            {.dtype = DataType::Float32(), .shape = StaticShape({3, 2})},
    };

    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingOp, InfersOutputShapeWithUint32Tokens) {
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::UInt(32), .shape = StaticShape({5})},
            {.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, PreservesSymbolicTokenAndHiddenDims) {
    constexpr EmbeddingParams params;
    const ShapeSymbol token_count = ShapeSymbol::Create();
    const ShapeSymbol hidden_size = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{token_count})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden_size})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Symbolic weight dims (vocab, hidden) emit DimPositiveConstraint for runtime validation.
    ASSERT_EQ(inference->runtime_checks.size(), 2U);
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 2U);
    EXPECT_EQ(inference->outputs[0].shape[0], token_count);
    EXPECT_EQ(inference->outputs[0].shape[1], hidden_size);
}

TEST(EmbeddingOp, InfersOutputShapeForRank2Tokens) {
    // Rank-2 token_ids [batch, seq] must produce [batch, seq, hidden],
    // preserving all input axes (PyTorch nn.Embedding semantics).
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({2, 5})},
            {.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    EXPECT_TRUE(inference->runtime_checks.empty());
    ASSERT_EQ(inference->outputs.size(), 1U);
    EXPECT_EQ(inference->outputs[0].dtype, DataType::Float32());
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0].GetStaticValue(), 2);
    EXPECT_EQ(inference->outputs[0].shape[1].GetStaticValue(), 5);
    EXPECT_EQ(inference->outputs[0].shape[2].GetStaticValue(), 4096);
}

TEST(EmbeddingOp, InfersOutputShapeForRank2SymbolicTokens) {
    // Symbolic rank-2 token_ids must preserve both symbolic axes in output.
    constexpr EmbeddingParams params;
    const ShapeSymbol batch = ShapeSymbol::Create();
    const ShapeSymbol seq = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{batch, seq})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), hidden})},
    };

    const StatusOr<InferenceResult> inference = InferOperator(OpType::kEmbedding, params, inputs);

    ASSERT_TRUE(inference.ok()) << inference.status().ToString();
    // Symbolic weight dims (vocab, hidden) emit DimPositiveConstraint for runtime validation.
    ASSERT_EQ(inference->runtime_checks.size(), 2U);
    ASSERT_EQ(inference->outputs.size(), 1U);
    ASSERT_EQ(inference->outputs[0].shape.rank(), 3U);
    EXPECT_EQ(inference->outputs[0].shape[0], batch);
    EXPECT_EQ(inference->outputs[0].shape[1], seq);
    EXPECT_EQ(inference->outputs[0].shape[2], hidden);
}

TEST(EmbeddingOp, AcceptsZeroTokenCount) {
    // Zero token count yields empty output [0, hidden]; valid NumPy-style embedding.
    constexpr EmbeddingParams params;
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({0})},
            {.dtype = DataType::Float32(), .shape = StaticShape({32000, 4096})},
    };
    EXPECT_TRUE(InferOperator(OpType::kEmbedding, params, inputs).status().ok());
}

TEST(EmbeddingOp, EmitsPositiveConstraintForSymbolicWeightDims) {
    // Symbolic weight dims must emit DimPositiveConstraint for runtime validation.
    constexpr EmbeddingParams params;
    const ShapeSymbol vocab = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const TensorSpec inputs[2] = {
            {.dtype = DataType::Int(64), .shape = StaticShape({3})},
            {.dtype = DataType::Float32(),
             .shape = SymbolicShape(std::vector<ShapeSymbol>{vocab, hidden})},
    };
    const auto result = InferOperator(OpType::kEmbedding, params, inputs);
    ASSERT_TRUE(result.ok()) << result.status().ToString();
    ASSERT_EQ(result->runtime_checks.size(), 2U);

    // Verify DimPositiveConstraint locators without relying on ordering.
    bool found_vocab = false;
    bool found_hidden = false;
    for (const auto& check: result->runtime_checks) {
        const auto* pos = std::get_if<DimPositiveConstraint>(&check.condition);
        if (pos == nullptr) continue;
        if (pos->dim.tensor_port.tensor_idx == 1U && pos->dim.dim_index == 0U) {
            found_vocab = true;
        } else if (pos->dim.tensor_port.tensor_idx == 1U && pos->dim.dim_index == 1U) {
            found_hidden = true;
        }
    }
    EXPECT_TRUE(found_vocab) << "missing DimPositiveConstraint for weight vocab_size (input[1] dim[0])";
    EXPECT_TRUE(found_hidden) << "missing DimPositiveConstraint for weight hidden_size (input[1] dim[1])";
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
    ::new (params_buffer) cpu::detail::EmbeddingParams{
            .token_ids = inputs[0],
            .weight = inputs[1],
            .output = outputs[0],
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
            .params_size = sizeof(cpu::detail::EmbeddingParams),
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
