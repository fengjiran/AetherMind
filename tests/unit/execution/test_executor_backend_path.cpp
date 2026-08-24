#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/cpu/cpu_workspace_arena.h"
#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan_builder.h"
#include "aethermind/execution/executor.h"
#include "aethermind/operators/operator_inference.h"
#include "aethermind/runtime/runtime_builder.h"

#include <gtest/gtest.h>

#include <vector>

namespace {

using namespace aethermind;

int g_kernel_calls = 0;
WorkspaceBinding g_workspace{};

Status RecordingKernel(const KernelContext& context) noexcept {
    ++g_kernel_calls;
    g_workspace = context.workspace_binding;
    return Status::Ok();
}

Status FailingKernel(const KernelContext&) noexcept {
    return Status::InvalidArgument("kernel failure");
}

class ExecutorTestBackend final : public Backend {
public:
    AM_NODISCARD DeviceType device_type() const noexcept override { return DeviceType::kCPU; }
    AM_NODISCARD const BackendCapabilities& capabilities() const noexcept override { return capabilities_; }

    StatusOr<ResolvedKernel> PrepareKernel(OpType op_type,
                                           const KernelSelector&,
                                           const OpParams&) const override {
        if (op_type == OpType::kSoftmax) {
            return ResolvedKernel{.op_type = op_type,
                                  .fn = &RecordingKernel,
                                  .workspace_requirement = {.bytes = 64, .alignment = 64}};
        }
        if (op_type == OpType::kArgmax) return ResolvedKernel{.op_type = op_type, .fn = &FailingKernel};
        if (op_type == OpType::kRmsNorm) return ResolvedKernel{.op_type = op_type, .fn = &RecordingKernel};
        return Status::NotFound("unsupported test kernel");
    }

    AM_NODISCARD const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }

private:
    BackendCapabilities capabilities_{};
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
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<ExecutorTestBackendFactory>());
    return builder.Build();
}

SymbolicShape StaticShape(std::initializer_list<int64_t> dimensions) {
    const std::vector<int64_t> copied(dimensions);
    return SymbolicShape(IntArrayView{copied});
}

ExecutionPlanNodeSpec MakeSoftmaxNode() {
    const TensorSpec spec{.dtype = DataType::Float32(), .shape = StaticShape({2, 4})};
    const std::vector<TensorSpec> inputs{spec};
    const auto analyzed = InferOperator(OpType::kSoftmax, OpParams{SoftmaxParams{.axis = -1}}, inputs);
    AM_CHECK(analyzed.ok());
    return {.op_type = OpType::kSoftmax,
            .selector = {.device_type = DeviceType::kCPU,
                         .act_dtype = DataType::Float32(),
                         .weight_dtype = DataType::Float32()},
            .input_specs = {spec},
            .output_specs = analyzed->outputs,
            .runtime_checks = analyzed->runtime_checks,
            .op_params = OpParams{SoftmaxParams{.axis = -1}}};
}

TEST(ExecutorBackendPath, ExecutesFrozenKernelWithCachedValueBindings) {
    RuntimeContext runtime = MakeRuntime();
    const auto plan = ExecutionPlanBuilder::Build(runtime, std::vector{MakeSoftmaxNode()});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    alignas(64) std::byte workspace[64]{};
    CpuWorkspaceArena arena(workspace, sizeof(workspace));
    float input[8]{};
    float output[8]{};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    const ExecutionStep& step = plan->steps().front();
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = step.inputs[0], .tensor = TensorView(input, DataType::Float32(), shape, strides)}},
             .writable = {{.value = step.outputs[0], .tensor = MutableTensorView(output, DataType::Float32(), shape, strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext bindings(&arena);
    bindings.SetBindingTable(std::move(*table));
    g_kernel_calls = 0;
    g_workspace = {};
    ASSERT_TRUE(Executor::Execute(*plan, bindings).ok());
    EXPECT_EQ(g_kernel_calls, 1);
    EXPECT_EQ(g_workspace.data, workspace);
    EXPECT_EQ(g_workspace.size, sizeof(workspace));
}

TEST(ExecutorBackendPath, PropagatesFrozenKernelFailure) {
    RuntimeContext runtime = MakeRuntime();
    const TensorSpec input_spec{.dtype = DataType::Float32(), .shape = StaticShape({2, 4})};
    const std::vector<TensorSpec> inputs{input_spec};
    const auto analyzed = InferOperator(OpType::kArgmax, OpParams{ArgmaxParams{.axis = -1}}, inputs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    const auto plan = ExecutionPlanBuilder::Build(
            runtime,
            std::vector{ExecutionPlanNodeSpec{.op_type = OpType::kArgmax,
                                              .selector = {.device_type = DeviceType::kCPU,
                                                           .act_dtype = DataType::Float32(),
                                                           .weight_dtype = DataType::Float32()},
                                              .input_specs = {input_spec},
                                              .output_specs = analyzed->outputs,
                                              .runtime_checks = analyzed->runtime_checks,
                                              .op_params = OpParams{ArgmaxParams{.axis = -1}}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    float input[8]{};
    int64_t output[2]{};
    const int64_t input_shape[2] = {2, 4};
    const int64_t input_strides[2] = {4, 1};
    const int64_t output_shape[1] = {2};
    const int64_t output_strides[1] = {1};
    const ExecutionStep& step = plan->steps().front();
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = step.inputs[0], .tensor = TensorView(input, DataType::Float32(), input_shape, input_strides)}},
             .writable = {{.value = step.outputs[0], .tensor = MutableTensorView(output, DataType::Int(64), output_shape, output_strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext bindings;
    bindings.SetBindingTable(std::move(*table));
    EXPECT_EQ(Executor::Execute(*plan, bindings).code(), StatusCode::kInvalidArgument);
}

TEST(ExecutorBackendPath, ExecuteFailsWhenWorkspaceRequirementCannotBeBound) {
    RuntimeContext runtime = MakeRuntime();
    const auto plan = ExecutionPlanBuilder::Build(runtime, std::vector{MakeSoftmaxNode()});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    float input[8]{};
    const int64_t shape[2] = {2, 4};
    const int64_t strides[2] = {4, 1};
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = plan->steps()[0].inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext bindings;
    bindings.SetBindingTable(std::move(*table));
    EXPECT_EQ(Executor::Execute(*plan, bindings).code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutorBackendPath, ExecuteRejectsViolatedRuntimeShapeConstraintBeforeRun) {
    RuntimeContext runtime = MakeRuntime();
    const ShapeSymbol sequence = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_length = ShapeSymbol::Create();
    const TensorSpec input_spec{.dtype = DataType::Float32(),
                                .shape = SymbolicShape({sequence, hidden})};
    const TensorSpec weight_spec{.dtype = DataType::Float32(),
                                 .shape = SymbolicShape({weight_length})};
    const std::vector<TensorSpec> input_specs{input_spec, weight_spec};
    const auto analyzed = InferOperator(OpType::kRmsNorm, OpParams{RmsNormParams{.eps = 1.0e-5F}}, input_specs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    const auto plan = ExecutionPlanBuilder::Build(
            runtime,
            std::vector{ExecutionPlanNodeSpec{.op_type = OpType::kRmsNorm,
                                              .selector = {.device_type = DeviceType::kCPU,
                                                           .act_dtype = DataType::Float32(),
                                                           .weight_dtype = DataType::Float32()},
                                              .input_specs = input_specs,
                                              .output_specs = analyzed->outputs,
                                              .runtime_checks = analyzed->runtime_checks,
                                              .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    float input[16]{};
    float weight[16]{};
    const int64_t input_shape[2] = {2, 8};
    const int64_t input_strides[2] = {8, 1};
    const int64_t weight_shape[1] = {16};
    const int64_t weight_strides[1] = {1};
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = plan->steps()[0].inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape, input_strides)},
                          {.value = plan->steps()[0].inputs[1],
                           .tensor = TensorView(weight, DataType::Float32(), weight_shape, weight_strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_FALSE(table.ok());
    EXPECT_EQ(table.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutorBackendPath, ExecuteRunsWhenRuntimeShapeConstraintIsSatisfied) {
    RuntimeContext runtime = MakeRuntime();
    const ShapeSymbol sequence = ShapeSymbol::Create();
    const ShapeSymbol hidden = ShapeSymbol::Create();
    const ShapeSymbol weight_length = ShapeSymbol::Create();
    const TensorSpec input_spec{.dtype = DataType::Float32(),
                                .shape = SymbolicShape({sequence, hidden})};
    const TensorSpec weight_spec{.dtype = DataType::Float32(),
                                 .shape = SymbolicShape({weight_length})};
    const std::vector<TensorSpec> input_specs{input_spec, weight_spec};
    const auto analyzed = InferOperator(OpType::kRmsNorm, OpParams{RmsNormParams{.eps = 1.0e-5F}}, input_specs);
    ASSERT_TRUE(analyzed.ok()) << analyzed.status().ToString();
    const auto plan = ExecutionPlanBuilder::Build(
            runtime,
            std::vector{ExecutionPlanNodeSpec{.op_type = OpType::kRmsNorm,
                                              .selector = {.device_type = DeviceType::kCPU,
                                                           .act_dtype = DataType::Float32(),
                                                           .weight_dtype = DataType::Float32()},
                                              .input_specs = input_specs,
                                              .output_specs = analyzed->outputs,
                                              .runtime_checks = analyzed->runtime_checks,
                                              .op_params = OpParams{RmsNormParams{.eps = 1.0e-5F}}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    float input[16]{};
    float weight[8]{};
    float output[16]{};
    const int64_t input_shape[2] = {2, 8};
    const int64_t input_strides[2] = {8, 1};
    const int64_t weight_shape[1] = {8};
    const int64_t weight_strides[1] = {1};
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = plan->steps()[0].inputs[0],
                           .tensor = TensorView(input, DataType::Float32(), input_shape, input_strides)},
                          {.value = plan->steps()[0].inputs[1],
                           .tensor = TensorView(weight, DataType::Float32(), weight_shape, weight_strides)}},
             .writable = {{.value = plan->steps()[0].outputs[0],
                           .tensor = MutableTensorView(output, DataType::Float32(), input_shape, input_strides)}}},
            runtime.GetAllocator(Device::CPU()));
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext bindings;
    bindings.SetBindingTable(std::move(*table));
    ASSERT_TRUE(Executor::Execute(*plan, bindings).ok());
}

}// namespace
