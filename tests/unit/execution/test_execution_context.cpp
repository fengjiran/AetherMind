#include "aethermind/execution/execution_context.h"
#include "aethermind/memory/cpu_allocator.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

class RecordingWorkspaceArena final : public WorkspaceArena {
public:
    explicit RecordingWorkspaceArena(WorkspaceBinding binding) noexcept
        : binding_(binding) {}

    WorkspaceBinding Bind(const WorkspaceRequirement& requirement) noexcept override {
        ++bind_calls_;
        last_requirement_ = requirement;
        return binding_;
    }

    void Reset() noexcept override {
        ++reset_calls_;
    }

    int bind_calls() const noexcept {
        return bind_calls_;
    }

    int reset_calls() const noexcept {
        return reset_calls_;
    }

    const WorkspaceRequirement& last_requirement() const noexcept {
        return last_requirement_;
    }

private:
    WorkspaceBinding binding_{};
    WorkspaceRequirement last_requirement_{};
    int bind_calls_ = 0;
    int reset_calls_ = 0;
};

StatusOr<ExecutionPlan> MakeEmptyPlan(WorkspacePlanLayout workspace_layout = {}) {
    return ExecutionPlan::Create({}, {}, {}, {}, {}, workspace_layout);
}

StatusOr<ExecutionContext> MakeContext(const ExecutionPlan& plan,
                                       CPUAllocator& allocator,
                                       WorkspaceArena* workspace_arena = nullptr) {
    auto prepared = PrepareExecutionBindings(plan, {}, allocator);
    if (!prepared.ok()) {
        return std::move(prepared).status();
    }
    return ExecutionContext::Create(plan, std::move(*prepared), workspace_arena);
}

Status NoopKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

TEST(ExecutionContext, ZeroByteWorkspaceDoesNotRequireArena) {
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    auto context = MakeContext(*plan, allocator);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const StatusOr<WorkspaceBinding> binding = context->BindWorkspace(
            WorkspaceRequirement{.bytes = 0, .alignment = 64, .offset = 0});

    ASSERT_TRUE(binding.ok());
    EXPECT_EQ(binding->data, nullptr);
    EXPECT_EQ(binding->size, 0U);
}

TEST(ExecutionContext, NonZeroWorkspaceWithoutArenaFails) {
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    auto context = MakeContext(*plan, allocator);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const StatusOr<WorkspaceBinding> binding = context->BindWorkspace(
            WorkspaceRequirement{.bytes = 32, .alignment = 32, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutionContext, BindWorkspaceDelegatesToArena) {
    alignas(64) std::byte storage[128]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    auto context = MakeContext(*plan, allocator, &arena);
    ASSERT_TRUE(context.ok()) << context.status().ToString();
    const WorkspaceRequirement requirement{.bytes = 64, .alignment = 64, .offset = 32};

    const StatusOr<WorkspaceBinding> binding = context->BindWorkspace(requirement);

    ASSERT_TRUE(binding.ok());
    EXPECT_EQ(binding->data, storage);
    EXPECT_EQ(binding->size, sizeof(storage));
    EXPECT_EQ(arena.bind_calls(), 1);
    EXPECT_EQ(arena.last_requirement().bytes, requirement.bytes);
    EXPECT_EQ(arena.last_requirement().alignment, requirement.alignment);
    EXPECT_EQ(arena.last_requirement().offset, requirement.offset);
}

TEST(ExecutionContext, BindWorkspaceRejectsInsufficientBinding) {
    alignas(64) std::byte storage[32]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    auto context = MakeContext(*plan, allocator, &arena);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const StatusOr<WorkspaceBinding> binding = context->BindWorkspace(
            WorkspaceRequirement{.bytes = 64, .alignment = 32, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kFailedPrecondition);
    EXPECT_EQ(arena.bind_calls(), 1);
}

TEST(ExecutionContext, CreateRejectsIncompatiblePreparedBindings) {
    const auto first_plan = MakeEmptyPlan();
    const auto second_plan = MakeEmptyPlan();
    ASSERT_TRUE(first_plan.ok()) << first_plan.status().ToString();
    ASSERT_TRUE(second_plan.ok()) << second_plan.status().ToString();

    CPUAllocator allocator(Device::CPU());
    auto prepared = PrepareExecutionBindings(*first_plan, {}, allocator);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    const auto context = ExecutionContext::Create(*second_plan, std::move(*prepared), nullptr);

    ASSERT_FALSE(context.ok());
    EXPECT_EQ(context.status().code(), StatusCode::kInvalidArgument);
}

TEST(ExecutionContext, CreateRequiresWorkspaceForPlanRequirement) {
    const auto plan = MakeEmptyPlan({.total_bytes = 64, .required_alignment = 64});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    CPUAllocator allocator(Device::CPU());
    auto prepared = PrepareExecutionBindings(*plan, {}, allocator);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    const auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);

    ASSERT_FALSE(context.ok());
    EXPECT_EQ(context.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutionContext, CreateRequiresKVViewForStateAliases) {
    const TensorSpec spec{
            .dtype = DataType::Float32(),
            .shape = SymbolicShape({ShapeSymbol::CreateFromValue(2)})};
    const auto plan = ExecutionPlan::Create(
            {{.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kModelInput},
             {.spec = spec, .kind = ExecutionValueKind::kState},
             {.spec = spec, .kind = ExecutionValueKind::kState},
             {.spec = spec, .kind = ExecutionValueKind::kState},
             {.spec = spec, .kind = ExecutionValueKind::kState}},
            {{.index = 0}, {.index = 1}}, {},
            {{.kernel = {.op_type = OpType::kKVCacheUpdate, .fn = &NoopKernel},
              .inputs = {{.index = 0}, {.index = 1}, {.index = 2}, {.index = 3}},
              .outputs = {{.index = 4}, {.index = 5}},
              .kernel_input_ports = {0, 1},
              .kernel_output_ports = {}}},
            {.aliases = {{.step_index = 0, .input_port = 2, .output_port = 0},
                         {.step_index = 0, .input_port = 3, .output_port = 1}}});
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    const float input[2]{};
    const int64_t shape[1] = {2};
    const int64_t strides[1] = {1};
    CPUAllocator allocator(Device::CPU());
    auto prepared = PrepareExecutionBindings(
            *plan,
            {.readable = {{.value = {.index = 0},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)},
                          {.value = {.index = 1},
                           .tensor = TensorView(input, DataType::Float32(), shape, strides)}}},
            allocator);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    const auto context = ExecutionContext::Create(*plan, std::move(*prepared), nullptr);

    EXPECT_FALSE(context.ok());
    EXPECT_EQ(context.status().code(), StatusCode::kFailedPrecondition);
}

TEST(ExecutionContext, ClearDoesNotResetBorrowedWorkspaceArena) {
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();

    CPUAllocator allocator(Device::CPU());
    auto prepared = PrepareExecutionBindings(*plan, {}, allocator);
    ASSERT_TRUE(prepared.ok()) << prepared.status().ToString();

    RecordingWorkspaceArena arena({});
    auto context = ExecutionContext::Create(*plan, std::move(*prepared), &arena);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    context->Clear();

    EXPECT_EQ(arena.reset_calls(), 0);
    EXPECT_EQ(context->prepared_bindings(), nullptr);
    EXPECT_EQ(context->workspace_arena(), nullptr);
    EXPECT_FALSE(context->HasKVCacheView());
}

TEST(ExecutionContext, InvalidAlignmentIsRejectedBeforeArenaBinding) {
    alignas(64) std::byte storage[64]{};
    RecordingWorkspaceArena arena(WorkspaceBinding{.data = storage, .size = sizeof(storage)});
    const auto plan = MakeEmptyPlan();
    ASSERT_TRUE(plan.ok()) << plan.status().ToString();
    CPUAllocator allocator(Device::CPU());
    auto context = MakeContext(*plan, allocator, &arena);
    ASSERT_TRUE(context.ok()) << context.status().ToString();

    const StatusOr<WorkspaceBinding> binding = context->BindWorkspace(
            WorkspaceRequirement{.bytes = 16, .alignment = 3, .offset = 0});

    ASSERT_FALSE(binding.ok());
    EXPECT_EQ(binding.status().code(), StatusCode::kInvalidArgument);
    EXPECT_EQ(arena.bind_calls(), 0);
}

} // namespace
