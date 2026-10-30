#include "aethermind/backend/kernel_context.h"
#include "aethermind/execution/execution_bindings.h"
#include "test_utils/test_resource_allocator.h"

#include "inference/test_malloc_interposer.h"
#include <array>
#include <gtest/gtest.h>

namespace {
using namespace aethermind;
Status NoOpResourceKernel(const KernelContext&) noexcept { return Status::Ok(); }
StatusOr<ExecutionPlan> MakeResourcePlan() {
    const std::array<int64_t, 1> dims{17};
    const TensorSpec spec{.dtype = DataType::Float32(), .shape = SymbolicShape(dims)};
    const ResolvedKernel kernel{.op_type = OpType::kSoftmax, .fn = &NoOpResourceKernel};
    return ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kActivation},
             {.spec = spec, .kind = ExecutionValueKind::kActivation}},
            {{.index = 0}}, {{.index = 2}},
            {{.selector = {.device_type = DeviceType::kCPU}, .kernel = kernel, .inputs = {{.index = 0}}, .outputs = {{.index = 1}}, .kernel_input_ports = {0}, .kernel_output_ports = {0}},
             {.selector = {.device_type = DeviceType::kCPU}, .kernel = kernel, .inputs = {{.index = 1}}, .outputs = {{.index = 2}}, .kernel_input_ports = {0}, .kernel_output_ports = {0}}});
}
StatusOr<ExecutionBindingPlan> PlanWithLocalMetadata(const ExecutionPlan& plan, const float* input) {
    const std::array<int64_t, 1> shape{17}, stride{1};
    ExternalTensorBindings external;
    external.read_only.push_back({{.index = 0}, TensorView(input, DataType::Float32(), shape, stride, alignof(float))});
    return PlanExecutionBindings(plan, external);
}
TEST(ExecutionBindingPlanning, PlansWithoutStorageAndMaterializesExactArena) {
    auto plan = MakeResourcePlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    std::array<float, 17> input{};
    auto layout = PlanWithLocalMetadata(*plan, input.data());
    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(layout->activation_bytes(), 196U);
    EXPECT_EQ(layout->activation_alignment(), 64U);
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    input[0] = 7;
    auto prepared = MaterializeExecutionBindings(*plan, std::move(*layout), allocator);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();
    ASSERT_EQ(stats->requests.size(), 1U);
    EXPECT_EQ(stats->requests[0].nbytes, 196U);
    EXPECT_EQ(prepared->values()[0].ReadView().shape()[0], 17);
    EXPECT_EQ(static_cast<const float*>(prepared->values()[0].ReadView().data())[0], 7);
    EXPECT_FALSE(layout->IsCompatible(*plan));
}
TEST(ExecutionBindingPlanning, RejectsWrongPlanAndMovedFromBeforeAllocation) {
    auto first = MakeResourcePlan(), second = MakeResourcePlan();
    ASSERT_TRUE(first.ok());
    ASSERT_TRUE(second.ok());
    std::array<float, 17> input{};
    auto layout = PlanWithLocalMetadata(*first, input.data());
    ASSERT_TRUE(layout.ok());
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    EXPECT_EQ(MaterializeExecutionBindings(*second, std::move(*layout), allocator).status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(MaterializeExecutionBindings(*first, std::move(*layout), allocator).status().code(), StatusCode::kFailedPrecondition);
    EXPECT_TRUE(stats->requests.empty());
}
TEST(ExecutionBindingPlanning, RejectsMissingAndDuplicateExternalBindings) {
    auto plan = MakeResourcePlan();
    ASSERT_TRUE(plan.ok());
    EXPECT_EQ(PlanExecutionBindings(*plan, {}).status().code(), StatusCode::kFailedPrecondition);
    std::array<float, 17> input{};
    const std::array<int64_t, 1> shape{17}, stride{1};
    ExternalTensorBindings external;
    auto binding = ExternalReadOnlyValueBinding{{.index = 0}, TensorView(input.data(), DataType::Float32(), shape, stride, 4)};
    external.read_only = {binding, binding};
    EXPECT_EQ(PlanExecutionBindings(*plan, external).status().code(), StatusCode::kInvalidArgument);
}
TEST(ExecutionBindingPlanning, WritableExternalOutputDoesNotConsumeArenaBytes) {
    auto plan = MakeResourcePlan();
    ASSERT_TRUE(plan.ok());
    std::array<float, 17> input{}, output{};
    const std::array<int64_t, 1> shape{17}, stride{1};
    ExternalTensorBindings external;
    external.read_only.push_back({{.index = 0}, TensorView(input.data(), DataType::Float32(), shape, stride, 4)});
    external.writable.push_back({{.index = 2}, MutableTensorView(output.data(), DataType::Float32(), shape, stride, 4)});
    auto layout = PlanExecutionBindings(*plan, external);
    ASSERT_TRUE(layout.ok());
    EXPECT_EQ(layout->activation_bytes(), 68U);
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    test::ResourceRecordingAllocator allocator(stats);
    auto prepared = MaterializeExecutionBindings(*plan, std::move(*layout), allocator);
    ASSERT_TRUE(prepared.ok());
    EXPECT_EQ(prepared->values()[2].ReadView().data(), output.data());
}
TEST(ExecutionBindingPlanning, StorageFailureLeavesPlanAndInputsUsableForRetry) {
    auto plan = MakeResourcePlan();
    ASSERT_TRUE(plan.ok());
    std::array<float, 17> input{};
    auto layout = PlanWithLocalMetadata(*plan, input.data());
    ASSERT_TRUE(layout.ok());
    auto stats = std::make_shared<test::ResourceAllocationStats>();
    stats->fail_on_call = 0;
    test::ResourceRecordingAllocator allocator(stats);
    EXPECT_EQ(MaterializeExecutionBindings(*plan, std::move(*layout), allocator).status().code(), StatusCode::kResourceExhausted);
    auto retry = PlanWithLocalMetadata(*plan, input.data());
    ASSERT_TRUE(retry.ok());
    EXPECT_TRUE(MaterializeExecutionBindings(*plan, std::move(*retry), allocator).ok());
}
TEST(ExecutionBindingPlanningDiagnosticFailure, MetadataFailureThrowsBeforePayloadAllocation) {
    if (!test::MallocInterposerAvailable()) GTEST_SKIP() << "Requires the glibc malloc interposer";
    auto plan = MakeResourcePlan();
    ASSERT_TRUE(plan.ok());
    std::array<float, 17> input{};
    const std::array<int64_t, 1> shape{17}, stride{1};
    ExternalTensorBindings external;
    external.read_only.push_back({{.index = 0}, TensorView(input.data(), DataType::Float32(), shape, stride, 4)});
    bool threw = false;
    test::ScopedMallocFailure failure(0);
    try {
        auto result = PlanExecutionBindings(*plan, external);
        UNUSED(result);
    } catch (const std::bad_alloc&) { threw = true; }
    const auto observed = failure.Stop();
    EXPECT_TRUE(observed.allocation_failed);
    EXPECT_TRUE(threw);
}
} // namespace
