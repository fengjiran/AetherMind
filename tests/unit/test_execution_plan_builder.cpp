#include "../../include/aethermind/execution/execution_plan_builder.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/backend/op_kernel_context.h"
#include "aethermind/backend/packed_weights.h"
#include "aethermind/memory/buffer.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <memory>
#include <span>

namespace aethermind {
namespace {

struct TestAttrs {
    int epsilon;
};

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

class TestPackedWeights final : public PackedWeights {
public:
    TestPackedWeights(OpType op_type,
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

Status PackedTestKernel(const KernelInvocation&,
                        const OpKernelContext&,
                        const WorkspaceBinding&) noexcept {
    return Status::Ok();
}

class PackedTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    const BackendCapabilities& capabilities() const noexcept override {
        return capabilities_;
    }

    KernelFunc ResolveKernel(OpType,
                             const KernelSelector& selector) const noexcept override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return nullptr;
        }
        return &PackedTestKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector& selector) const noexcept override {
        if (selector.weight_format != WeightFormat::kPacked) {
            return Status::NotFound("Packed test backend only resolves packed selectors");
        }
        return ResolvedKernel{
                .op_type = op_type,
                .fn = &PackedTestKernel,
                .attrs = {},
                .debug_name = "test::packed_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities capabilities_{};
};

class PackedTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override {
        return DeviceType::kCPU;
    }

    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<PackedTestBackend>();
    }
};

ExecutionPlanNodeSpec MakeRmsNormNodeSpec(std::span<const std::byte> attrs = {}) {
    return ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .attrs = attrs,
    };
}

TEST(ExecutionPlanBuilder, ResolveKernelForNodeUsesOpTypeDirectly) {
    CpuBackend backend;
    TestAttrs attrs{.epsilon = 42};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       MakeRmsNormNodeSpec(attrs_bytes));

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ(resolved->op_type, OpType::kRMSNorm);
    ASSERT_NE(resolved->fn, nullptr);
    EXPECT_EQ(resolved->attrs.data(), attrs_bytes.data());
    EXPECT_EQ(resolved->attrs.size(), sizeof(attrs));
    EXPECT_STREQ(resolved->debug_name, "test::fake_cpu_kernel");
}

TEST(ExecutionPlanBuilder, BuildFreezesResolvedKernelIntoExecutionPlan) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();
    TestAttrs attrs{.epsilon = 11};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(MakeRmsNormNodeSpec(attrs_bytes));

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->size(), 1U);

    attrs.epsilon = 99;

    const auto& step = plan->steps().front();
    ASSERT_EQ(step.attrs.size(), sizeof(TestAttrs));
    const auto* stored_attrs = reinterpret_cast<const TestAttrs*>(step.attrs.data());
    ASSERT_NE(stored_attrs, nullptr);
    EXPECT_EQ(step.op_type, OpType::kRMSNorm);
    EXPECT_EQ(step.invocation.op_type, OpType::kRMSNorm);
    EXPECT_EQ(step.invocation.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(step.packed_params, nullptr);
    EXPECT_EQ(step.workspace_requirement.bytes, 0U);
    EXPECT_EQ(step.workspace_requirement.alignment, 64U);
    EXPECT_EQ(step.workspace_requirement.offset, 0U);
    EXPECT_EQ(stored_attrs->epsilon, 11);
    EXPECT_STREQ(step.debug_name, "test::fake_cpu_kernel");
}

TEST(ExecutionPlanBuilder, BuildPlansWorkspaceOffsetsAcrossNodes) {
    RuntimeBuilder builder;
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
            .workspace_requirement = {.bytes = 32, .alignment = 16, .offset = 999},
    });
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
            .workspace_requirement = {.bytes = 8, .alignment = 64, .offset = 123},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 2U);
    EXPECT_EQ(plan->steps()[0].workspace_requirement.offset, 0U);
    EXPECT_EQ(plan->steps()[1].workspace_requirement.offset, 64U);
}

TEST(ExecutionPlanBuilder, BuildBindsPackedParamsFromModelInstanceSidecar) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<PackedTestBackendFactory>());
    RuntimeContext runtime = builder.Build();
    ModelInstance model_instance;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };

    ASSERT_TRUE(model_instance.StorePackedWeights(std::make_unique<TestPackedWeights>(
                                                          OpType::kRMSNorm,
                                                          selector,
                                                          MakeTestBuffer(128)))
                        .ok());

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, model_instance, nodes);

    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);
    ASSERT_NE(plan->steps().front().packed_params, nullptr);
    EXPECT_EQ(plan->steps().front().packed_params,
              model_instance.FindPackedWeights(OpType::kRMSNorm, selector)->storage().data());
}

TEST(ExecutionPlanBuilder, BuildRejectsPackedWeightNodeWithoutModelInstanceSidecar) {
    RuntimeBuilder builder;
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kRMSNorm,
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPacked,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kNotFound);
}

TEST(ExecutionPlanBuilder, ResolveKernelForNodeRejectsUnknownOpType) {
    CpuBackend backend;

    const StatusOr<ResolvedKernel> resolved =
            ExecutionPlanBuilder::ResolveKernelForNode(backend,
                                                       ExecutionPlanNodeSpec{
                                                               .op_type = OpType::kUnknown,
                                                               .device_type = DeviceType::kCPU,
                                                               .activation_dtype = DataType::Float32(),
                                                               .weight_dtype = DataType::Float32(),
                                                       });

    EXPECT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
