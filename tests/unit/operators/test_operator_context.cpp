#include "aethermind/backend/cpu/cpu_backend.h"
#include "aethermind/execution/workspace_arena.h"
#include "aethermind/operators/operator_context.h"

#include <gtest/gtest.h>

namespace aethermind {
namespace {

class TestWorkspaceArena final : public WorkspaceArena {
public:
    WorkspaceBinding Bind(const WorkspaceRequirement&) noexcept override {
        return {};
    }

    void Reset() noexcept override {}
};

TEST(Operators_OperatorContext, DefaultsAreNullAndGeneric) {
    OperatorContext ctx;

    EXPECT_EQ(ctx.backend, nullptr);
    EXPECT_EQ(ctx.kernel_registry, nullptr);
    EXPECT_EQ(ctx.workspace, nullptr);
    EXPECT_EQ(ctx.selector.device_type, DeviceType::kUndefined);
    EXPECT_EQ(ctx.selector.phase, ExecPhase::kBoth);
    EXPECT_EQ(ctx.phase(), ExecPhase::kBoth);
    EXPECT_FALSE(ctx.enable_profiling);
    EXPECT_FALSE(ctx.enable_debug_check);
}

TEST(Operators_OperatorContext, CanCarryRuntimeDependenciesAndKernelSelector) {
    CpuBackend backend;
    TestWorkspaceArena workspace;
    KernelSelector selector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kDecode,
    };

    OperatorContext ctx{
            .backend = &backend,
            .kernel_registry = backend.TryGetKernelRegistryForDebug(),
            .workspace = &workspace,
            .selector = selector,
            .enable_profiling = true,
            .enable_debug_check = true,
    };

    EXPECT_EQ(ctx.backend, &backend);
    EXPECT_EQ(ctx.kernel_registry, backend.TryGetKernelRegistryForDebug());
    EXPECT_EQ(ctx.workspace, &workspace);
    EXPECT_EQ(ctx.selector, selector);
    EXPECT_EQ(ctx.phase(), ExecPhase::kDecode);
    EXPECT_TRUE(ctx.enable_profiling);
    EXPECT_TRUE(ctx.enable_debug_check);
}

}// namespace
}// namespace aethermind
