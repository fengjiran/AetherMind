#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/op_kernel_context.h"
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

namespace aethermind {
namespace {

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

Status ImmutableKernel(const KernelInvocation&,
                       const OpKernelContext&,
                       const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

class ImmutablePackedWeights final : public PackedWeights {
public:
    ImmutablePackedWeights(OpType op_type,
                           KernelSelector selector,
                           Buffer storage,
                           bool* destroyed_flag) noexcept
        : op_type_(op_type),
          selector_(selector),
          storage_(std::move(storage)),
          destroyed_flag_(destroyed_flag) {}

    ~ImmutablePackedWeights() override {
        if (destroyed_flag_ != nullptr) {
            *destroyed_flag_ = true;
        }
    }

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
    bool* destroyed_flag_ = nullptr;
};

class ImmutableTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    KernelFunc ResolveKernel(OpType,
                             const KernelSelector&) const noexcept override {
        return &ImmutableKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector&) const noexcept override {
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &ImmutableKernel,
                .attrs = {},
                .debug_name = "test::immutable_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class ImmutableTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<ImmutableTestBackend>();
    }
};

TEST(ExecutionPlanImmutability, StepsReturnsConstViewAfterConstruction) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const std::vector<ExecutionStep>& steps = plan->steps();
    EXPECT_EQ(steps.size(), 1U);
    EXPECT_EQ(steps.front().op_type, OpType::kRMSNorm);
    EXPECT_NE(steps.front().fn, nullptr);
}

TEST(ExecutionPlanImmutability, WorkspaceOffsetsAreFrozenAfterBuilderPlanning) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {.bytes = 64, .alignment = 32, .offset = 999},
    });
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {.bytes = 128, .alignment = 64, .offset = 123},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 2U);

    const ExecutionStep& step0 = plan->steps()[0];
    const ExecutionStep& step1 = plan->steps()[1];

    EXPECT_EQ(step0.workspace_requirement.offset, 0U);
    EXPECT_EQ(step1.workspace_requirement.offset, 64U);

    EXPECT_EQ(step0.workspace_requirement.bytes, 64U);
    EXPECT_EQ(step1.workspace_requirement.bytes, 128U);
}

TEST(ExecutionPlanImmutability, PackedParamsLifetimeManagedByModelInstance) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    bool packed_destroyed = false;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };

    ModelInstance model_instance;
    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<ImmutablePackedWeights>(
                                                          OpType::kRMSNorm,
                                                          selector,
                                                          MakeTestBuffer(256),
                                                          &packed_destroyed))
                        .ok());

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, model_instance, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const ExecutionStep& step = plan->steps().front();
    ASSERT_NE(step.packed_params, nullptr);

    const void* packed_ptr = step.packed_params;
    EXPECT_FALSE(packed_destroyed);

    {
        ExecutionPlan local_plan = plan.value();
        const ExecutionStep& local_step = local_plan.steps().front();
        EXPECT_EQ(local_step.packed_params, packed_ptr);
    }

    EXPECT_FALSE(packed_destroyed);
}

TEST(ExecutionPlanImmutability, ExecutorConsumesFrozenPlanWithoutModification) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());

    const size_t original_size = plan->size();
    const OpType original_op_type = plan->steps().front().op_type;
    const KernelFunc original_fn = plan->steps().front().fn;

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(*plan, bindings);

    ASSERT_TRUE(status.ok());

    EXPECT_EQ(plan->size(), original_size);
    EXPECT_EQ(plan->steps().front().op_type, original_op_type);
    EXPECT_EQ(plan->steps().front().fn, original_fn);
}

TEST(ExecutionPlanImmutability, PlanDoesNotContainRuntimeBindings) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<ImmutableTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);

    const ExecutionStep& step = plan->steps().front();

    EXPECT_EQ(step.packed_params, nullptr);

    EXPECT_GT(step.workspace_requirement.alignment, 0U);
    EXPECT_TRUE((step.workspace_requirement.alignment &
                 (step.workspace_requirement.alignment - 1)) == 0);
}

}// namespace
}// namespace aethermind
