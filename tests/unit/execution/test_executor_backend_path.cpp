#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/operators/operator_registry.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace aethermind {
namespace {

std::vector<int>* g_execution_order = nullptr;
int* g_constraint_operator_runs = nullptr;
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

    explicit RuntimeTensorStorage(std::vector<int64_t> input_shape)
        : shape(std::move(input_shape)),
          strides(MakeStrides(shape)) {}

    AM_NODISCARD TensorView View() const {
        return {nullptr, DataType::Float32(), shape, strides};
    }
};

class RuntimeConstraintTestOperator final : public Operator {
public:
    using Params = AttentionParams;

    explicit RuntimeConstraintTestOperator(Params) noexcept {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kAttention;
    }

    AM_NODISCARD Status ValidateParams() const override {
        return Status::Ok();
    }

    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec>) const override {
        return Status::Ok();
    }

    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec>) const override {
        return InferenceResult{
                .outputs = {},
                .runtime_checks = {ShapeConstraint{
                        .condition = DimEqualConstraint{
                                .lhs = DimLocator{.tensor_port = {.direction = TensorPortType::kInput,
                                                                  .tensor_idx = 0},
                                                  .dim_index = 1},
                                .rhs = DimLocator{.tensor_port = {.direction = TensorPortType::kInput,
                                                                  .tensor_idx = 1},
                                                  .dim_index = 0}},
                        .error_context = "runtime hidden size mismatch",
                }},
        };
    }

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override {
        if (ctx.backend == nullptr) {
            return Status::InvalidArgument("RuntimeConstraintTestOperator backend is null");
        }
        const auto resolved = ctx.backend->ResolveKernelInfo(Type(), ctx.selector);
        if (!resolved.ok()) {
            return resolved.status();
        }
        resolved_ = resolved.value();
        return Status::Ok();
    }

    AM_NODISCARD Status Run(KernelContext&,
                            const RuntimeBindingContext&,
                            size_t) const noexcept override {
        if (g_constraint_operator_runs != nullptr) {
            ++(*g_constraint_operator_runs);
        }
        return Status::Ok();
    }

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_;
    }

private:
    ResolvedKernel resolved_{};
};

const bool kRuntimeConstraintTestOperatorRegistered = OperatorRegistry::RegisterOrAbort(
        OpType::kAttention,
        OperatorRegistry::Descriptor{
                .factory_ = &OperatorRegistry::CreateTypedOperator<RuntimeConstraintTestOperator>,
                .make_default_params_ = []() -> StatusOr<OpParams> {
                    return OpParams{RuntimeConstraintTestOperator::Params{}};
                },
        },
        "RuntimeConstraintTestOperator");

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

    AM_NODISCARD KernelFunc ResolveKernel(OpType op_type, const KernelSelector&) const noexcept override {
        switch (op_type) {
            case OpType::kSilu:
                return &FirstKernel;
            case OpType::kRoPE:
                return &SecondKernel;
            case OpType::kMatMul:
                return &FailingKernel;
            case OpType::kAttention:
                return &FirstKernel;
            default:
                return nullptr;
        }
    }

    AM_NODISCARD StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector&) const noexcept override {
        KernelFunc fn = ResolveKernel(op_type, KernelSelector{});
        if (fn == nullptr) {
            return Status::NotFound("ExecutorTestBackend does not resolve this op type");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = fn,
                .attrs = {},
                .debug_name = GetDebugName(op_type),
        };
    }

    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    static const char* GetDebugName(OpType op_type) noexcept {
        switch (op_type) {
            case OpType::kSilu:
                return "test::first_kernel";
            case OpType::kRoPE:
                return "test::second_kernel";
            case OpType::kMatMul:
                return "test::failing_kernel";
            case OpType::kAttention:
                return "test::runtime_constraint_kernel";
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
    alignas(64) std::byte workspace[256]{};
    CpuWorkspaceArena arena(workspace, sizeof(workspace));
    RuntimeBindingContext bindings(&arena);
    g_execution_order = &execution_order;
    ModelInstance model_instance;

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
    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<ExecutorPackedWeights>(
                                                          OpType::kRoPE,
                                                          packed_selector,
                                                          std::move(packed_storage)))
                        .ok());

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kSilu,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 64,
                    .alignment = 64,
            },
    });
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRoPE,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .workspace_requirement = {
                    .bytes = 128,
                    .alignment = 64,
            },
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, model_instance, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 2U);

    const Status status = Executor::Execute(*plan, bindings);

    g_execution_order = nullptr;
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(execution_order, (std::vector<int>{1, 2}));
    EXPECT_EQ(g_last_kernel_context.workspace, &arena);
    EXPECT_EQ(g_last_kernel_context.workspace_binding.size, 128U);
    EXPECT_EQ(g_last_kernel_context.workspace_binding.data,
              static_cast<void*>(workspace + 64));
    EXPECT_EQ(g_last_kernel_context.device_type, DeviceType::kCPU);
    EXPECT_EQ(g_last_kernel_context.packed_weights, expected_packed_weights);
    EXPECT_EQ(g_last_kernel_context.kernel_params, nullptr);
}

TEST(ExecutorBackendPath, ExecutePropagatesKernelFailure) {
    RuntimeContext runtime = MakeRuntime();
    alignas(32) std::byte workspace[64]{};
    CpuWorkspaceArena arena(workspace, sizeof(workspace));
    RuntimeBindingContext bindings(&arena);

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kMatMul,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 32,
                    .alignment = 32,
            },
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const Status status = Executor::Execute(*plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(ExecutorBackendPath, ExecuteFailsWhenWorkspaceRequirementCannotBeBound) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kSilu,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 32,
                    .alignment = 32,
            },
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const Status status = Executor::Execute(*plan, bindings);

    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutorBackendPath, ExecuteRejectsViolatedRuntimeShapeConstraintBeforeRun) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;
    int run_count = 0;
    g_constraint_operator_runs = &run_count;

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kAttention,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = {
                    TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), ShapeSymbol::Create()})},
                    TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create()})},
            },
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_EQ(plan->steps().front().runtime_checks.size(), 1U);

    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight{std::vector<int64_t>{16}};
    bindings.SetStepTensorBinding(0, StepTensorBinding{.inputs = {input.View(), weight.View()}, .outputs = {}});

    const Status status = Executor::Execute(*plan, bindings);

    g_constraint_operator_runs = nullptr;
    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(status.message(), "runtime hidden size mismatch");
    EXPECT_EQ(run_count, 0);
}

TEST(ExecutorBackendPath, ExecuteRunsWhenRuntimeShapeConstraintIsSatisfied) {
    RuntimeContext runtime = MakeRuntime();
    RuntimeBindingContext bindings;
    int run_count = 0;
    g_constraint_operator_runs = &run_count;

    ExecutionPlanNodeSpec node{
            .op_type = OpType::kAttention,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = {
                    TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create(), ShapeSymbol::Create()})},
                    TensorSpec{.dtype = DataType::Float32(), .shape = SymbolicShape(std::vector<ShapeSymbol>{ShapeSymbol::Create()})},
            },
    };

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, std::vector<ExecutionPlanNodeSpec>{node});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    RuntimeTensorStorage input{std::vector<int64_t>{2, 8}};
    RuntimeTensorStorage weight{std::vector<int64_t>{8}};
    bindings.SetStepTensorBinding(0, StepTensorBinding{.inputs = {input.View(), weight.View()}, .outputs = {}});

    const Status status = Executor::Execute(*plan, bindings);

    g_constraint_operator_runs = nullptr;
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(run_count, 1);
}

}// namespace
}// namespace aethermind
