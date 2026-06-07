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
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <vector>

namespace aethermind {
namespace {

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

class ExecutorPackedWeights final : public PackedWeights {
public:
    ExecutorPackedWeights(OpType op_type,
                          KernelSelector selector,
                          Buffer storage) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)) {}

    OpType op_type() const noexcept override {
        return op_type_;
    }

    const KernelSelector& selector() const noexcept override {
        return selector_;
    }

    const Buffer& storage() const noexcept override {
        return storage_;
    }

private:
    OpType op_type_ = OpType::kUnknown;
    KernelSelector selector_{};
    Buffer storage_{};
};

class ExecutorTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return caps_; }

    KernelFunc ResolveKernel(OpType op_type, const KernelSelector&) const noexcept override {
        switch (op_type) {
            case OpType::kLinear:
                return &FirstKernel;
            case OpType::kRoPE:
                return &SecondKernel;
            case OpType::kMatMul:
                return &FailingKernel;
            default:
                return nullptr;
        }
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
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

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    static const char* GetDebugName(OpType op_type) noexcept {
        switch (op_type) {
            case OpType::kLinear:
                return "test::first_kernel";
            case OpType::kRoPE:
                return "test::second_kernel";
            case OpType::kMatMul:
                return "test::failing_kernel";
            default:
                return "test::unknown_kernel";
        }
    }

    BackendCapabilities caps_{};
};

class ExecutorTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    std::unique_ptr<Backend> Create() const override {
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
            .activation_dtype = DataType::Float32(),
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
            .op_type = OpType::kLinear,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 64,
                    .alignment = 64,
            },
    });
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRoPE,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
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
            .activation_dtype = DataType::Float32(),
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
            .op_type = OpType::kLinear,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
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

}  // namespace
}  // namespace aethermind
