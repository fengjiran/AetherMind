#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/workspace_types.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <memory>

namespace aethermind {
namespace {

class ResolveCounters {
public:
    void IncrementResolveKernelCalls() noexcept {
        ++resolve_kernel_calls_;
    }

    void IncrementResolveKernelInfoCalls() noexcept {
        ++resolve_kernel_info_calls_;
    }

    void IncrementKernelInvocations() noexcept {
        ++kernel_invocations_;
    }

    AM_NODISCARD int resolve_kernel_calls() const noexcept {
        return resolve_kernel_calls_;
    }

    AM_NODISCARD int resolve_kernel_info_calls() const noexcept {
        return resolve_kernel_info_calls_;
    }

    AM_NODISCARD int kernel_invocations() const noexcept {
        return kernel_invocations_;
    }

private:
    int resolve_kernel_calls_ = 0;
    int resolve_kernel_info_calls_ = 0;
    int kernel_invocations_ = 0;
};

ResolveCounters* g_resolve_counters = nullptr;

Status CountingKernel(const KernelInvocation&,
                      const OpKernelContext&,
                      const WorkspaceBinding&) noexcept {
    if (g_resolve_counters != nullptr) {
        g_resolve_counters->IncrementKernelInvocations();
    }
    return Status::Ok();
}

class CountingBackend final : public Backend {
public:
    explicit CountingBackend(std::shared_ptr<ResolveCounters> counters)
        : counters_(std::move(counters)) {}

    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    KernelFunc ResolveKernel(OpType,
                             const KernelSelector&) const noexcept override {
        counters_->IncrementResolveKernelCalls();
        return &CountingKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector&) const noexcept override {
        counters_->IncrementResolveKernelInfoCalls();
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &CountingKernel,
                .attrs = {},
                .debug_name = "test::counting_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    std::shared_ptr<ResolveCounters> counters_;
    BackendCapabilities capabilities_{};
};

class CountingBackendFactory final : public BackendFactory {
public:
    explicit CountingBackendFactory(std::shared_ptr<ResolveCounters> counters)
        : counters_(std::move(counters)) {}

    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<CountingBackend>(counters_);
    }

private:
    std::shared_ptr<ResolveCounters> counters_;
};

TEST(NoHotpathResolve, ExecutorConsumesFrozenKernelWithoutBackendLookup) {
    auto counters = std::make_shared<ResolveCounters>();
    g_resolve_counters = counters.get();

    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<CountingBackendFactory>(counters));
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    EXPECT_EQ(counters->resolve_kernel_info_calls(), 1);
    EXPECT_EQ(counters->resolve_kernel_calls(), 0);
    EXPECT_EQ(counters->kernel_invocations(), 0);

    RuntimeBindingContext bindings;
    const Status status = Executor::Execute(*plan, bindings);

    g_resolve_counters = nullptr;
    ASSERT_TRUE(status.ok());
    EXPECT_EQ(counters->resolve_kernel_info_calls(), 1);
    EXPECT_EQ(counters->resolve_kernel_calls(), 0);
    EXPECT_EQ(counters->kernel_invocations(), 1);
}

}// namespace
}// namespace aethermind
