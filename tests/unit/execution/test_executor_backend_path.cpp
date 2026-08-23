#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/model/packed_weight_store.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

// Over-aligned scratch storage sized and aligned from a plan workspace layout.
struct AlignedScratch {
    explicit AlignedScratch(size_t bytes, size_t alignment)
        : bytes(bytes),
          alignment(alignment),
          data(static_cast<std::byte*>(
                  ::operator new(bytes, std::align_val_t{alignment}))) {}
    ~AlignedScratch() {
        ::operator delete(data, std::align_val_t{alignment});
    }
    AlignedScratch(const AlignedScratch&) = delete;
    AlignedScratch& operator=(const AlignedScratch&) = delete;

    size_t bytes = 0;
    size_t alignment = 0;
    std::byte* data = nullptr;
};

std::vector<int>* g_execution_order = nullptr;
KernelContext g_last_kernel_context{};

Status FirstKernel(const KernelContext& ctx) noexcept {
    g_last_kernel_context = ctx;
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(1);
    }
    return Status::Ok();
}

Status SecondKernel(const KernelContext& ctx) noexcept {
    g_last_kernel_context = ctx;
    if (g_execution_order != nullptr) {
        g_execution_order->push_back(2);
    }
    return Status::Ok();
}

Status FailingKernel(const KernelContext&) noexcept {
    return Status::InvalidArgument("kernel failure");
}

void FreeTestBuffer(void*, void* ptr) noexcept {
    std::free(ptr);
}

Buffer MakeTestBuffer(size_t nbytes, size_t alignment = 64) {
    void* ptr = nullptr;
    const int rc = posix_memalign(&ptr, alignment, nbytes == 0 ? 1 : nbytes);
    if (rc != 0 || ptr == nullptr) {
        return {};
    }
    return Buffer{nbytes, MemoryHandle(ptr, nullptr, &FreeTestBuffer, Device::CPU(), alignment)};
}

std::vector<int64_t> MakeStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size(), 1);
    for (int64_t i = static_cast<int64_t>(shape.size()) - 2; i >= 0; --i) {
        strides[static_cast<size_t>(i)] = strides[static_cast<size_t>(i + 1)] * shape[static_cast<size_t>(i + 1)];
    }
    return strides;
}

struct RuntimeTensorStorage {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
    std::vector<float> storage;

    explicit RuntimeTensorStorage(std::vector<int64_t> input_shape)
        : shape(std::move(input_shape)),
          strides(MakeStrides(shape)) {
        size_t numel = 1;
        for (const auto d: shape) {
            numel *= static_cast<size_t>(std::max<int64_t>(d, 0));
        }
        storage.resize(numel, 0.0F);
    }

    AM_NODISCARD TensorView View() const {
        return {storage.data(), DataType::Float32(), shape, strides};
    }

    AM_NODISCARD MutableTensorView MutableView() {
        return {storage.data(), DataType::Float32(), shape, strides};
    }
};

// Helper: derive RmsNorm output_specs and runtime_checks via InferOperator.
StatusOr<InferenceResult> InferRmsNorm(float eps,
                                       const SymbolicShape& act_shape,
                                       const SymbolicShape& weight_shape) {
    std::vector<TensorSpec> inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    return InferOperator(OpType::kRmsNorm,
                         OpParams{RmsNormParams{.eps = eps}},
                         inputs);
}

class ExecutorPackedWeights final : public PackedWeights {
public:
    ExecutorPackedWeights(OpType op_type,
                          KernelSelector selector,
                          Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)) {}

    AM_NODISCARD OpType op_type() const noexcept override {
        return op_type_;
    }

    AM_NODISCARD const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    AM_NODISCARD const Buffer& storage() const noexcept override {
        return storage_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
};

class ExecutorTestBackend final : public Backend {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override { return caps_; }

    AM_NODISCARD StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams&) const override {
        KernelFunc fn = nullptr;
        WorkspaceRequirement workspace_requirement{};
        switch (op_type) {
            case OpType::kSoftmax:
                workspace_requirement = selector.phase == ExecPhase::kPrefill
                                                ? WorkspaceRequirement{.bytes = 64, .alignment = 64}
                                        : selector.phase == ExecPhase::kDecode
                                                ? WorkspaceRequirement{.bytes = 32, .alignment = 32}
                                                : WorkspaceRequirement{};
                fn = &FirstKernel;
                break;
            case OpType::kAttention:
            case OpType::kRmsNorm:
                fn = &FirstKernel;
                break;
            case OpType::kRoPE:
                fn = &SecondKernel;
                workspace_requirement = {.bytes = 128, .alignment = 64};
                break;
            case OpType::kArgmax:
                fn = &FailingKernel;
                workspace_requirement = {.bytes = 32, .alignment = 32};
                break;
            default:
                break;
        }
        if (fn == nullptr) {
            return Status::NotFound("ExecutorTestBackend does not prepare this op type");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = fn,
                .attrs = {},
                .debug_name = GetDebugName(op_type),
                .workspace_requirement = workspace_requirement,
        };
    }

    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    static const char* GetDebugName(OpType op_type) noexcept {
        switch (op_type) {
            case OpType::kSoftmax:
                return "test::first_kernel";
            case OpType::kRoPE:
                return "test::second_kernel";
            case OpType::kArgmax:
                return "test::failing_kernel";
            case OpType::kAttention:
                return "test::runtime_constraint_kernel";
            case OpType::kRmsNorm:
                return "test::rmsnorm_constraint_kernel";
            default:
                return "test::unknown_kernel";
        }
    }

    BackendCapabilities caps_{};
};

class ExecutorTestBackendFactory final : public BackendFactory {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    AM_NODISCARD std::unique_ptr<Backend> Create() const override {
        return std::make_unique<ExecutorTestBackend>();
    }
};

RuntimeContext MakeRuntime() {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ExecutorTestBackendFactory>());
    return builder.Build();
}

TEST(ExecutorBackendPath, ExecuteRunsFrozenKernelsInPlanOrder) {
    RuntimeContext runtime = MakeRuntime();
    std::vector<int> execution_order;
    g_execution_order = &execution_order;
    PackedWeightStore packed_weight_store;

    const KernelSelector packed_selector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
    Buffer packed_storage = MakeTestBuffer(64);
    const void* expected_packed_weights = packed_storage.data();
    ASSERT_NE(expected_packed_weights, nullptr);
    ASSERT_TRUE(packed_weight_store.Store(std::make_unique<ExecutorPackedWeights>(
                                                  OpType::kRoPE,
                                                  packed_selector,
                                                  std::move(packed_storage)))
                        .ok());

    // kSoftmax is schema-only in this test backend. InferSoftmax expects one
    // float32 input and echoes it.
    const SymbolicShape softmax_in_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{4, 8}});
    std::vector<TensorSpec> softmax_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = softmax_in_shape},
    };
    const auto softmax_analyzed = InferOperator(
            OpType::kSoftmax, OpParams{SoftmaxParams{.axis = -1}}, softmax_inputs);
    ASSERT_TRUE(softmax_analyzed.ok()) << softmax_analyzed.status().ToString();

    // kRoPE is also resolved directly from the test backend. InferRoPE
    // expects q/k rank-2 float32 [seq_len, hidden], position_ids rank-1
    // int64 [seq_len]; hidden widths must equal num_*_heads * head_dim.
    const SymbolicShape rope_q_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{2, 8}});
    const SymbolicShape rope_k_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{2, 8}});
    const SymbolicShape rope_pos_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{2}});
    std::vector<TensorSpec> rope_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = rope_q_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = rope_k_shape},
            TensorSpec{.dtype = DataType::Int(64), .shape = rope_pos_shape},
    };
    const RoPEParams rope_params{
            .head_dim = 2,
            .num_attention_heads = 4,
            .num_key_value_heads = 4,
            .max_position_embeddings = 128,
            .theta = 10000.0,
    };
    const auto rope_analyzed = InferOperator(
            OpType::kRoPE, OpParams{rope_params}, rope_inputs);
    ASSERT_TRUE(rope_analyzed.ok()) << rope_analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    ExecutionPlanNodeSpec softmax_node{
            .op_type = OpType::kSoftmax,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .phase = ExecPhase::kPrefill,
            },
            .workspace_requirement = {.bytes = 64, .alignment = 64},
    };
    softmax_node.op_params = OpParams{SoftmaxParams{.axis = -1}};
    softmax_node.input_specs = softmax_inputs;
    softmax_node.output_specs = softmax_analyzed->outputs;
    softmax_node.runtime_checks = softmax_analyzed->runtime_checks;
    nodes.push_back(std::move(softmax_node));

    ExecutionPlanNodeSpec rope_node{
            .op_type = OpType::kRoPE,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPacked,
            },
            .workspace_requirement = {.bytes = 128, .alignment = 64},
    };
    rope_node.op_params = OpParams{rope_params};
    rope_node.input_specs = rope_inputs;
    rope_node.output_specs = rope_analyzed->outputs;
    rope_node.runtime_checks = rope_analyzed->runtime_checks;
    nodes.push_back(std::move(rope_node));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, packed_weight_store, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 2U);

    const size_t workspace_bytes = plan->total_workspace_bytes();
    const size_t workspace_alignment = plan->workspace_alignment();
    EXPECT_EQ(workspace_bytes, 192U);
    EXPECT_EQ(workspace_alignment, 64U);
    AlignedScratch scratch(workspace_bytes, workspace_alignment);
    CpuWorkspaceArena arena(scratch.data, workspace_bytes);
    RuntimeBindingContext bindings(&arena);

    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {TensorView{}},
                                             .outputs = {MutableTensorView{}},
                                     });
    bindings.SetStepTensorBinding(1, StepTensorBinding{
                                             .inputs = {TensorView{}, TensorView{}, TensorView{}},
                                             .outputs = {MutableTensorView{}, MutableTensorView{}},
                                     });

    const Status status = Executor::Execute(*plan, bindings);

    g_execution_order = nullptr;
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(execution_order, (std::vector<int>{1, 2}));
    EXPECT_EQ(g_last_kernel_context.workspace, &arena);
    EXPECT_EQ(g_last_kernel_context.workspace_binding.size, 128U);
    EXPECT_EQ(g_last_kernel_context.workspace_binding.data,
              static_cast<void*>(scratch.data + 64));
    EXPECT_EQ(g_last_kernel_context.device_type, DeviceType::kCPU);
    EXPECT_EQ(g_last_kernel_context.packed_weights, expected_packed_weights);
    EXPECT_EQ(g_last_kernel_context.kernel_params, nullptr);
}

TEST(ExecutorBackendPath, ExecutePropagatesKernelFailure) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;

    // kArgmax is resolved directly from the test backend. InferArgmax expects
    // one float32 input and ArgmaxParams.
    const SymbolicShape argmax_in_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{4, 8}});
    std::vector<TensorSpec> argmax_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = argmax_in_shape},
    };
    const auto argmax_analyzed = InferOperator(
            OpType::kArgmax, OpParams{ArgmaxParams{.axis = -1}}, argmax_inputs);
    ASSERT_TRUE(argmax_analyzed.ok()) << argmax_analyzed.status().ToString();

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kArgmax,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
            },
            .workspace_requirement = {.bytes = 32, .alignment = 32},
    };
    node.op_params = OpParams{ArgmaxParams{.axis = -1}};
    node.input_specs = argmax_inputs;
    node.output_specs = argmax_analyzed->outputs;
    node.runtime_checks = argmax_analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    const size_t workspace_bytes = plan->total_workspace_bytes();
    const size_t workspace_alignment = plan->workspace_alignment();
    EXPECT_EQ(workspace_bytes, 32U);
    EXPECT_EQ(workspace_alignment, 32U);
    AlignedScratch scratch(workspace_bytes, workspace_alignment);
    CpuWorkspaceArena arena(scratch.data, workspace_bytes);
    bindings.SetWorkspaceArena(&arena);

    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {TensorView{}},
                                             .outputs = {MutableTensorView{}},
                                     });

    const Status status = Executor::Execute(*plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ExecutorBackendPath, ExecuteFailsWhenWorkspaceRequirementCannotBeBound) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;

    const SymbolicShape softmax_in_shape = SymbolicShape(IntArrayView{std::vector<int64_t>{4, 8}});
    std::vector<TensorSpec> softmax_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = softmax_in_shape},
    };
    const auto softmax_analyzed = InferOperator(
            OpType::kSoftmax, OpParams{SoftmaxParams{.axis = -1}}, softmax_inputs);
    ASSERT_TRUE(softmax_analyzed.ok()) << softmax_analyzed.status().ToString();

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kSoftmax,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .phase = ExecPhase::kDecode,
            },
            .workspace_requirement = {.bytes = 32, .alignment = 32},
    };
    node.op_params = OpParams{SoftmaxParams{.axis = -1}};
    node.input_specs = softmax_inputs;
    node.output_specs = softmax_analyzed->outputs;
    node.runtime_checks = softmax_analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    const Status status = Executor::Execute(*plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutorBackendPath, ExecuteRejectsViolatedRuntimeShapeConstraintBeforeRun) {
    // InferAttention now emits a DimPositiveConstraint for symbolic seq_len,
    // but its q.hidden / cache.kv_heads / cache.head_dim must be static in
    // Phase 1, limiting the constraint variants it can produce. Switch to
    // kRmsNorm: InferRmsNorm produces exactly the DimEqualConstraint
    // (input[0].dim[1] == input[1].dim[0]) the original kAttention test
    // intended, when hidden and weight_len are distinct symbolic dimensions.
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;
    std::vector<int> execution_order;
    g_execution_order = &execution_order;

    const ShapeSymbol seq = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_dim = ShapeSymbol::Create();
    const SymbolicShape act_shape(std::vector<ShapeSymbol>{seq, hidden});
    const SymbolicShape weight_shape(std::vector<ShapeSymbol>{weight_dim});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    ASSERT_GE(analyzed->runtime_checks.size(), 1U);

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
            },
    };
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_GE(plan->steps().front().runtime_checks.size(), 1U);

    // Runtime shapes violate the constraint: input[0].dim[1]=8 != input[1].dim[0]=16.
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight{std::vector<int64_t>{16}};
    RuntimeTensorStorage output{std::vector<int64_t>{2, 8}};
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {input.View(), weight.View()},
                                             .outputs = {output.MutableView()},
                                     });

    const Status status = Executor::Execute(*plan, bindings);

    g_execution_order = nullptr;
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "RmsNorm hidden dimension must match weight length");
    EXPECT_TRUE(execution_order.empty());
}

TEST(ExecutorBackendPath, ExecuteRunsWhenRuntimeShapeConstraintIsSatisfied) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;
    std::vector<int> execution_order;
    g_execution_order = &execution_order;

    const ShapeSymbol seq = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_dim = ShapeSymbol::Create();
    const SymbolicShape act_shape(std::vector<ShapeSymbol>{seq, hidden});
    const SymbolicShape weight_shape(std::vector<ShapeSymbol>{weight_dim});
    const auto analyzed = InferRmsNorm(1.0e-5F, act_shape, weight_shape);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kRmsNorm,
            .selector = {
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
            },
    };
    node.op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}};
    node.input_specs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = act_shape},
            TensorSpec{.dtype = DataType::Float32(), .shape = weight_shape},
    };
    node.output_specs = analyzed->outputs;
    node.runtime_checks = analyzed->runtime_checks;

    const StatusOr<ExecutionPlan> plan =
            ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    // Runtime shapes satisfy the constraint: input[0].dim[1]=8 == input[1].dim[0]=8.
    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight{std::vector<int64_t>{8}};
    RuntimeTensorStorage output{std::vector<int64_t>{2, 8}};
    bindings.SetStepTensorBinding(0, StepTensorBinding{
                                             .inputs = {input.View(), weight.View()},
                                             .outputs = {output.MutableView()},
                                     });

    const Status status = Executor::Execute(*plan, bindings);

    g_execution_order = nullptr;
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(execution_order.size(), 1U);
}

}// namespace
