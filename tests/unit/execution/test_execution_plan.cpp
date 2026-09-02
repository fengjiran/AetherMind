#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_bindings.h"
#include "aethermind/execution/execution_plan.h"
#include "aethermind/execution/executor.h"
#include "aethermind/execution/kernel_invoker.h"
#include "aethermind/execution/runtime_binding_context.h"
#include "aethermind/memory/cpu_allocator.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

int g_invocations = 0;
int g_built_param = 0;

Status FakeKernel(const KernelContext&) noexcept {
    ++g_invocations;
    return Status::Ok();
}

Status CaptureBuiltParamKernel(const KernelContext& context) noexcept {
    ++g_invocations;
    if (context.kernel_params == nullptr) return Status::FailedPrecondition("missing params");
    g_built_param = *static_cast<const int*>(context.kernel_params);
    return Status::Ok();
}

Status BuildKernelParam(std::span<const TensorView>,
                        std::span<const MutableTensorView>,
                        void* params) noexcept {
    *static_cast<int*>(params) = 17;
    return Status::Ok();
}

Status FailToBuildKernelParam(std::span<const TensorView>,
                              std::span<const MutableTensorView>,
                              void*) noexcept {
    return Status::InvalidArgument("test builder failure");
}

class CountingAllocator final : public Allocator {
public:
    CountingAllocator() : allocator_(Device::CPU()) {}

    Buffer Allocate(size_t bytes) override {
        ++allocation_count_;
        return allocator_.Allocate(bytes);
    }

    AM_NODISCARD Device device() const noexcept override {
        return allocator_.device();
    }

    AM_NODISCARD size_t allocation_count() const noexcept {
        return allocation_count_;
    }

private:
    CPUAllocator allocator_;
    size_t allocation_count_ = 0;
};

TensorSpec FloatVectorSpec(int64_t size) {
    return {.dtype = DataType::Float32(),
            .shape = SymbolicShape(IntArrayView{std::vector<int64_t>{size}})};
}

StatusOr<ExecutionPlan> MakeSingleSoftmaxPlan(
        ResolvedKernel kernel = {.op_type = OpType::kSoftmax, .fn = &FakeKernel}) {
    const TensorSpec spec = FloatVectorSpec(2);
    return ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput, .name = "input"},
             {.spec = spec, .kind = ExecutionValueKind::kActivation, .name = "output"}},
            {{.index = 0}}, {{.index = 1}},
            {{.selector = {.device_type = DeviceType::kCPU},
              .kernel = std::move(kernel),
              .inputs = {{.index = 0}},
              .outputs = {{.index = 1}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}});
}

StatusOr<ExecutionPlan> MakeSingleSoftmaxPlanForSpec(const TensorSpec& spec) {
    const ResolvedKernel kernel{.op_type = OpType::kSoftmax, .fn = &FakeKernel};
    return ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput, .name = "input"},
             {.spec = spec, .kind = ExecutionValueKind::kActivation, .name = "output"}},
            {{.index = 0}}, {{.index = 1}},
            {{.selector = {.device_type = DeviceType::kCPU},
              .kernel = kernel,
              .inputs = {{.index = 0}},
              .outputs = {{.index = 1}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}});
}

StatusOr<ExecutionPlan> MakeRmsNormPlan(std::vector<ShapeConstraint> runtime_checks) {
    const TensorSpec act_spec{.dtype = DataType::Float32(), .shape = SymbolicShape(IntArrayView{std::vector<int64_t>{2, 4}})};
    const TensorSpec weight_spec{.dtype = DataType::Float32(), .shape = SymbolicShape(IntArrayView{std::vector<int64_t>{4}})};
    return ExecutionPlan::Create(
            {{.spec = act_spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = weight_spec, .kind = ExecutionValueKind::kWeight},
             {.spec = act_spec, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}}, {{.index = 2}},
            {{.kernel = {.op_type = OpType::kRmsNorm, .fn = &FakeKernel},
              .inputs = {{.index = 0}, {.index = 1}},
              .outputs = {{.index = 2}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {0},
              .runtime_checks = std::move(runtime_checks)}});
}

TEST(ExecutionPlan, OwnsLogicalDataflowAndModelIo) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    ASSERT_EQ(plan->values().size(), 2U);
    ASSERT_EQ(plan->model_inputs().size(), 1U);
    ASSERT_EQ(plan->model_outputs().size(), 1U);
    ASSERT_EQ(plan->steps().size(), 1U);
    EXPECT_EQ(plan->steps()[0].inputs[0].index, 0U);
    EXPECT_EQ(plan->steps()[0].outputs[0].index, 1U);
    EXPECT_EQ(plan->steps()[0].kernel_input_ports, std::vector<uint32_t>{0});
    EXPECT_EQ(plan->steps()[0].kernel_output_ports, std::vector<uint32_t>{0});
}

TEST(ExecutionPlan, StoresResolvedKernelByValue) {
    ResolvedKernel kernel{.op_type = OpType::kSoftmax,
                          .fn = &FakeKernel,
                          .attrs = {std::byte{7}},
                          .name = "test::fake_kernel"};
    const auto plan = MakeSingleSoftmaxPlan(kernel);
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    EXPECT_EQ(plan->steps()[0].kernel.fn, &FakeKernel);
    EXPECT_EQ(plan->steps()[0].kernel.attrs, kernel.attrs);
    EXPECT_NE(plan->steps()[0].kernel.attrs.data(), kernel.attrs.data());
}

TEST(ExecutionPlan, RejectsInvalidResolvedKernel) {
    EXPECT_EQ(ExecutionPlan::Create({}, {}, {}, {ExecutionStep{}}).status().code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(ExecutionPlan::Create({}, {}, {},
                                    {ExecutionStep{.kernel = {.op_type = OpType::kSoftmax,
                                                              .fn = &FakeKernel,
                                                              .params_size = 1}}})
                      .status()
                      .code(),
              StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, AcceptsRuntimeCheckWithinCompactSpecRanges) {
    const ShapeConstraint check{
            .condition = DimEqualConstraint{.lhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 0}, .dim_index = 1},
                                            .rhs = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1}, .dim_index = 0}},
            .error_context = "hidden mismatch"};
    EXPECT_TRUE(MakeRmsNormPlan({check}).ok());
}

TEST(ExecutionPlan, RejectsRuntimeCheckReferencingMissingTensorPort) {
    const ShapeConstraint check{
            .condition = RankEqualConstraint{.port = {.direction = TensorPortType::kInput, .tensor_idx = 2}, .target_rank = 2},
            .error_context = "unused"};
    const auto plan = MakeRmsNormPlan({check});
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, RejectsRuntimeCheckReferencingDimensionBeyondRank) {
    const ShapeConstraint check{
            .condition = DimPositiveConstraint{.dim = {.tensor_port = {.direction = TensorPortType::kInput, .tensor_idx = 1}, .dim_index = 1}},
            .error_context = "unused"};
    const auto plan = MakeRmsNormPlan({check});
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(KernelInvoker, BuildsParamsOnlyForTheKernelCall) {
    g_invocations = 0;
    g_built_param = 0;
    KernelContext context{};
    ASSERT_TRUE(InvokeKernel({.op_type = OpType::kAdd,
                              .fn = &CaptureBuiltParamKernel,
                              .params_builder = &BuildKernelParam,
                              .params_size = sizeof(int)},
                             context, {}, {})
                        .ok());
    EXPECT_EQ(g_invocations, 1);
    EXPECT_EQ(g_built_param, 17);
}

TEST(KernelInvoker, PropagatesBuilderFailureWithoutCallingKernel) {
    g_invocations = 0;
    KernelContext context{};
    EXPECT_EQ(InvokeKernel({.op_type = OpType::kAdd,
                            .fn = &CaptureBuiltParamKernel,
                            .params_builder = &FailToBuildKernelParam,
                            .params_size = sizeof(int)},
                           context, {}, {})
                      .code(),
              StatusCode::kInvalidArgument);
    EXPECT_EQ(g_invocations, 0);
}

TEST(KernelInvoker, RejectsNullFunctionAndInvalidParamsContract) {
    KernelContext context{};
    EXPECT_EQ(InvokeKernel({.op_type = OpType::kAdd}, context, {}, {}).code(),
              StatusCode::kFailedPrecondition);
    EXPECT_EQ(InvokeKernel({.op_type = OpType::kAdd, .fn = &FakeKernel, .params_size = 1}, context, {}, {}).code(),
              StatusCode::kFailedPrecondition);
}

TEST(ExecutionPlan, RejectsActivationInputWithoutEarlierProducer) {
    const TensorSpec spec = FloatVectorSpec(2);
    const ResolvedKernel kernel{.op_type = OpType::kSoftmax, .fn = &FakeKernel};
    const auto plan = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kActivation}}, {}, {},
            {{.kernel = kernel,
              .inputs = {{.index = 0}},
              .outputs = {{.index = 0}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}});
    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, RejectsModelInputValueMissingFromModelInputs) {
    const TensorSpec spec = FloatVectorSpec(2);
    const auto plan = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput}},
            {}, {}, {});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, RejectsActivationWithoutProducer) {
    const TensorSpec spec = FloatVectorSpec(2);
    const auto plan = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kActivation}},
            {}, {{.index = 0}}, {});

    ASSERT_FALSE(plan.ok());
    EXPECT_EQ(plan.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, ExecuteRejectsBindingDTypeDrift) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    constexpr float input[2]{};
    constexpr int64_t shape[1] = {2};
    constexpr int64_t strides[1] = {1};
    CPUAllocator allocator(Device::CPU());
    const auto bindings = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float(16), shape, strides)}}},
            allocator);
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, ExecuteRejectsBindingRankDrift) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    constexpr float input[2]{};
    constexpr int64_t shape[2] = {1, 2};
    constexpr int64_t strides[2] = {2, 1};
    CPUAllocator allocator(Device::CPU());
    const auto bindings = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, ExecuteRejectsBindingStaticDimDrift) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    constexpr float input[3]{};
    constexpr int64_t shape[1] = {3};
    constexpr int64_t strides[1] = {1};
    CPUAllocator allocator(Device::CPU());
    const auto bindings = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, ExecuteRejectsInvalidViewsWhenCheckingPremises) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    const auto bindings = BuildExecutionBindings(
            *plan, ExternalValueBindings{.readable = {{.value = {.index = 0}, .tensor = TensorView{}}}}, allocator);
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionPlan, SortsStateAliasPlanForStepLookup) {
    const TensorSpec tensor_spec = FloatVectorSpec(2);
    const ResolvedKernel kernel{.op_type = OpType::kKVCacheUpdate, .fn = &FakeKernel};
    const auto plan = ExecutionPlan::Create(
            {{.spec = tensor_spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = tensor_spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = tensor_spec, .kind = ExecutionValueKind::kState},
             {.spec = tensor_spec, .kind = ExecutionValueKind::kState},
             {.spec = tensor_spec, .kind = ExecutionValueKind::kState},
             {.spec = tensor_spec, .kind = ExecutionValueKind::kState}},
            {{.index = 0}, {.index = 1}}, {},
            {{.kernel = kernel,
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 4}, {.index = 5}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {}}},
            {.aliases = {{.step_index = 0, .input_port = 3, .output_port = 1},
                         {.step_index = 0, .input_port = 2, .output_port = 0}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const auto aliases = plan->state_alias_plan().ForStep(0);
    ASSERT_EQ(aliases.size(), 2U);
    EXPECT_EQ(aliases[0].input_port, 3U);
    EXPECT_EQ(aliases[1].input_port, 2U);
}

TEST(ExecutionPlan, RejectsStateAliasBeyondStepCount) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const TensorSpec spec = FloatVectorSpec(2);
    const auto invalid = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}}, {{.index = 1}},
            {{.kernel = {.op_type = OpType::kSoftmax, .fn = &FakeKernel},
              .inputs = {{.index = 0}},
              .outputs = {{.index = 1}},
              .kernel_input_ports = {0},
              .kernel_output_ports = {0}}},
            {.aliases = {{.step_index = 1}}});
    ASSERT_FALSE(invalid.ok());
    EXPECT_EQ(invalid.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionBindings, BuildsCanonicalActivationStorageAndSurvivesMove) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const float input[2] = {1.0F, 2.0F};
    int64_t shape[1] = {2};
    int64_t strides[1] = {1};
    CountingAllocator allocator;
    auto bindings = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = plan->model_inputs()[0],
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_TRUE(bindings.ok()) << bindings.status().ToString();
    shape[0] = 1;
    strides[0] = 0;
    BindingTable moved = std::move(*bindings);
    ASSERT_EQ(moved.step_count(), 1U);
    ASSERT_EQ(moved.values().size(), 2U);
    EXPECT_TRUE(moved.values()[1].readable.is_valid());
    EXPECT_TRUE(moved.values()[1].writable.is_valid());
    EXPECT_EQ(moved.step(0).inputs[0].data(), input);
    EXPECT_EQ(moved.step(0).inputs[0].shape()[0], 2);
    EXPECT_EQ(moved.step(0).inputs[0].strides()[0], 1);
    RuntimeBindingContext context;
    context.SetBindingTable(std::move(moved));
    g_invocations = 0;
    ASSERT_TRUE(Executor::Execute(*plan, context).ok());
    EXPECT_EQ(g_invocations, 1);
}

TEST(ExecutionBindings, RejectsExternalDtypeAndSymbolIdentityDrift) {
    const ShapeSymbol symbol = ShapeSymbol::Create();
    const TensorSpec spec{.dtype = DataType::Float32(), .shape = SymbolicShape({symbol})};
    const ResolvedKernel kernel{.op_type = OpType::kAdd, .fn = &FakeKernel};
    const auto plan = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}, {.index = 1}}, {{.index = 2}},
            {{.kernel = kernel,
              .inputs = {{.index = 0}, {.index = 1}},
              .outputs = {{.index = 2}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {0}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const float lhs[2]{};
    const float rhs[3]{};
    const int64_t lhs_shape[1] = {2};
    const int64_t rhs_shape[1] = {3};
    const int64_t strides[1] = {1};
    CPUAllocator allocator(Device::CPU());
    const auto bindings = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0}, .tensor = TensorView(lhs, DataType::Float32(), lhs_shape, strides)},
                          {.value = {.index = 1}, .tensor = TensorView(rhs, DataType::Float32(), rhs_shape, strides)}}},
            allocator);
    ASSERT_FALSE(bindings.ok());
    EXPECT_EQ(bindings.status().code(), StatusCode::kInvalidArgument);
}

TEST(Executor, RunsCachedBindingTableWithoutPerStepBindingApi) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const float input[2]{};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    CountingAllocator allocator;
    auto table = BuildExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    RuntimeBindingContext context;
    context.SetBindingTable(std::move(*table));
    const size_t allocations_before_execute = allocator.allocation_count();
    g_invocations = 0;
    const Status status = Executor::Execute(*plan, context);
    ASSERT_TRUE(status.ok()) << status.ToString();
    EXPECT_EQ(g_invocations, 1);
    EXPECT_EQ(allocator.allocation_count(), allocations_before_execute);
}

TEST(BindingTable, RejectsDifferentPlanWithMatchingStepCountAndAcceptsPlanCopy) {
    const auto first_plan = MakeSingleSoftmaxPlan();
    const auto second_plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(first_plan.ok()) << first_plan.status().ToString();
    ASSERT_TRUE(second_plan.ok()) << second_plan.status().ToString();
    ASSERT_NE(first_plan->binding_key().value, 0U);
    ASSERT_NE(second_plan->binding_key().value, 0U);
    ASSERT_NE(first_plan->binding_key(), second_plan->binding_key());
    const ExecutionPlan copied_plan = *first_plan;

    const float input[2]{};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    CountingAllocator allocator;
    auto table = BuildExecutionBindings(
            *first_plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_TRUE(table.ok()) << table.status().ToString();
    EXPECT_TRUE(table->IsCompatible(*first_plan));
    EXPECT_TRUE(table->IsCompatible(copied_plan));
    EXPECT_FALSE(table->IsCompatible(*second_plan));

    RuntimeBindingContext context;
    context.SetBindingTable(std::move(*table));
    EXPECT_EQ(Executor::Execute(*second_plan, context).code(), StatusCode::kInvalidArgument);
}

TEST(RuntimeBindingContext, ResetAllowsReinstallingBindingTable) {
    const auto plan = MakeSingleSoftmaxPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    const float input[2]{};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    CountingAllocator allocator;
    const ExternalValueBindings external{
            .readable = {{.value = {.index = 0},
                          .tensor = TensorView(input, DataType::Float32(), shape, strides)}}};
    auto first_table = BuildExecutionBindings(*plan, external, allocator);
    ASSERT_TRUE(first_table.ok()) << first_table.status().ToString();
    RuntimeBindingContext context;
    context.SetBindingTable(std::move(*first_table));
    ASSERT_NE(context.binding_table(), nullptr);

    context.Reset();
    EXPECT_EQ(context.binding_table(), nullptr);

    auto second_table = BuildExecutionBindings(*plan, external, allocator);
    ASSERT_TRUE(second_table.ok()) << second_table.status().ToString();
    context.SetBindingTable(std::move(*second_table));
    EXPECT_NE(context.binding_table(), nullptr);
}

} // namespace
