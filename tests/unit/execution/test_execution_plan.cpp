#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>
#include <span>

namespace aethermind {
namespace {

SymbolicShape StaticShape(std::initializer_list<int64_t> dims) {
    const std::vector<int64_t> shape(dims);
    return SymbolicShape(IntArrayView{shape});
}

struct TestAttrs {
    int epsilon;
    int axis;
};

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

class StubTestBackend final : public Backend {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    const BackendCapabilities& capabilities() const noexcept override { return caps_; }

    KernelFunc ResolveKernel(OpType, const KernelSelector&) const noexcept override {
        return &FakeKernel;
    }

    StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector&) const noexcept override {
        return ResolvedKernel{
                .op_type = op_type,
                .fn = op_type == OpType::kArgmax ? nullptr : &FakeKernel,
                .attrs = {},
                .debug_name = "test::stub_kernel",
        };
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override {
        return nullptr;
    }

private:
    BackendCapabilities caps_{};
};

class StubTestBackendFactory final : public BackendFactory {
public:
    DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    std::unique_ptr<Backend> Create() const override {
        return std::make_unique<StubTestBackend>();
    }
};

TEST(ExecutionPlan, BuildFreezesOperatorResolvedAttrs) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<StubTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    TestAttrs attrs{.epsilon = 7, .axis = 3};
    const auto attrs_bytes = std::as_bytes(std::span{&attrs, size_t{1}});

    const SymbolicShape softmax_shape = StaticShape({2, 3});
    std::vector<TensorSpec> softmax_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = softmax_shape},
    };
    const auto analyzed = InferOperator(OpType::kSoftmax,
                                        OpParams{SoftmaxParams{.axis = -1}},
                                        softmax_inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kSoftmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 128,
                    .alignment = 64,
            },
            .input_specs = softmax_inputs,
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .attrs = std::vector<std::byte>(attrs_bytes.begin(), attrs_bytes.end()),
            .op_params = OpParams{SoftmaxParams{.axis = -1}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());

    attrs.epsilon = 99;

    ASSERT_EQ(plan->size(), 1U);
    const auto& step = plan->steps().front();
    const ResolvedKernel resolved = step.op->GetResolvedKernel();
    ASSERT_EQ(resolved.attrs.size(), sizeof(TestAttrs));
    const auto* stored_attrs = reinterpret_cast<const TestAttrs*>(resolved.attrs.data());
    ASSERT_NE(stored_attrs, nullptr);
    EXPECT_NE(stored_attrs, &attrs);
    EXPECT_EQ(step.op->Type(), OpType::kSoftmax);
    EXPECT_EQ(step.selector.device_type, DeviceType::kCPU);
    EXPECT_EQ(step.packed_weights, nullptr);
    EXPECT_EQ(step.workspace_requirement.bytes, 128U);
    EXPECT_EQ(step.workspace_requirement.alignment, 64U);
    EXPECT_EQ(step.workspace_requirement.offset, 0U);
    EXPECT_EQ(stored_attrs->epsilon, 7);
    EXPECT_EQ(stored_attrs->axis, 3);
    EXPECT_STREQ(resolved.debug_name, "test::stub_kernel");
}

TEST(ExecutionPlan, BuildAllowsEmptyAttrs) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<StubTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    const SymbolicShape softmax_shape = StaticShape({2, 3});
    std::vector<TensorSpec> softmax_inputs = {
            TensorSpec{.dtype = DataType::Float32(), .shape = softmax_shape},
    };
    const auto analyzed = InferOperator(OpType::kSoftmax,
                                        OpParams{SoftmaxParams{.axis = -1}},
                                        softmax_inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kSoftmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .input_specs = softmax_inputs,
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .attrs = {},
            .op_params = OpParams{SoftmaxParams{.axis = -1}},
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    ASSERT_TRUE(plan.ok());
    ASSERT_EQ(plan->size(), 1U);
    EXPECT_TRUE(plan->steps().front().op->GetResolvedKernel().attrs.empty());
    EXPECT_EQ(plan->steps().front().packed_weights, nullptr);
}

TEST(ExecutionPlan, BuildRejectsInvalidWorkspaceAlignment) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<StubTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kSoftmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .workspace_requirement = {
                    .bytes = 64,
                    .alignment = 24,
            },
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    EXPECT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, BuildRejectsNullResolvedKernelFunction) {
    RuntimeBuilder builder;
    builder.RegisterBackendFactory(DeviceType::kCPU,
                                   std::make_unique<StubTestBackendFactory>());
    RuntimeContext runtime = builder.Build();

    std::vector<ExecutionPlanNodeSpec> nodes;
    nodes.push_back(ExecutionPlanNodeSpec{
            .op_type = OpType::kArgmax,
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
    });

    const StatusOr<ExecutionPlan> plan = ExecutionPlanBuilder::Build(runtime, nodes);
    EXPECT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

}// namespace
}// namespace aethermind
