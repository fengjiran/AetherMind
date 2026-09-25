#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/runtime/workspace.h"

#include "aethermind/dtypes/data_type.h"

#include <gtest/gtest.h>

#include <string_view>
#include <thread>
#include <vector>

namespace {
using namespace aethermind;

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

KernelDef MakeTestKernelDef() {
    return KernelDef{
            .op_type = OpType::kRmsNorm,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .act_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .phase = ExecPhase::kBoth,
            },
            .kernel_func = &FakeKernel,
            .priority = 1,
            .name = "test::op",
    };
}

KernelSelector MakeMissingSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .act_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .phase = ExecPhase::kBoth,
    };
}

TEST(KernelRegistry, PackedDefRequiresValidRecipeAlignment) {
    KernelDef descriptor = MakeTestKernelDef();
    descriptor.selector.weight_format = WeightFormat::kPacked;
    descriptor.packing_recipe = PackingRecipe{
            .layout = "test_layout",
            .alignment = alignof(void*) / 2U,
    };
    EXPECT_EQ(ValidateKernelDef(descriptor).code(),
              StatusCode::kInvalidArgument);

    descriptor.packing_recipe.alignment = alignof(void*);
    EXPECT_TRUE(ValidateKernelDef(descriptor).ok());

    descriptor.selector.weight_format = WeightFormat::kPlain;
    EXPECT_EQ(ValidateKernelDef(descriptor).code(),
              StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, FindCandidatesReturnsStructuralMatches) {
    KernelRegistry registry;
    const KernelDef descriptor = MakeTestKernelDef();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(OpType::kRmsNorm, descriptor.selector);
    ASSERT_TRUE(candidates.ok());
    ASSERT_EQ(candidates->size(), 1U);
    EXPECT_EQ((*candidates)[0]->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, FindCandidatesReturnsEmptyForMissingKey) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(OpType::kLinear, MakeMissingSelector());
    ASSERT_TRUE(candidates.ok());
    EXPECT_TRUE(candidates->empty());
}

TEST(KernelRegistry, FindCandidatesRejectsUnknownOpType) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(OpType::kUnknown, MakeMissingSelector());

    ASSERT_FALSE(candidates.ok());
    EXPECT_EQ(candidates.status().code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, FindCandidatesBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDef descriptor = MakeTestKernelDef();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const auto candidates = registry.FindCandidates(OpType::kRmsNorm, descriptor.selector);
    EXPECT_EQ(candidates.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, FindByOpTypeBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDef descriptor = MakeTestKernelDef();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const auto found = registry.FindByOpType(OpType::kRmsNorm);
    EXPECT_EQ(found.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, RegisterAfterFreezeFails) {
    KernelRegistry registry;
    const KernelDef descriptor = MakeTestKernelDef();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());
    EXPECT_TRUE(registry.frozen());

    KernelDef extra = descriptor;
    extra.op_type = OpType::kLinear;
    extra.name = "test::other_op";

    const Status status = registry.Register(extra);
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, DuplicateRegistrationFails) {
    KernelRegistry registry;
    const KernelDef descriptor = MakeTestKernelDef();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    EXPECT_EQ(registry.Register(descriptor).code(), StatusCode::kAlreadyExists);
}

TEST(KernelRegistry, RegisterRejectsEmptyKernelName) {
    KernelRegistry registry;
    KernelDef descriptor = MakeTestKernelDef();
    descriptor.name = {};

    const Status status = registry.Register(descriptor);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, RegisterRejectsCpuRequirementsForNonCpuKernel) {
    KernelRegistry registry;
    KernelDef descriptor = MakeTestKernelDef();
    descriptor.selector.device_type = DeviceType::kCUDA;
    descriptor.cpu_requirements = CpuFeatureSet::From({CpuFeature::kAvx2});

    const Status status = registry.Register(descriptor);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, FindByOpTypeReturnsMatchingDefs) {
    KernelRegistry registry;
    const KernelDef rms = MakeTestKernelDef();

    KernelDef linear = rms;
    linear.op_type = OpType::kLinear;
    linear.name = "test::linear";

    ASSERT_TRUE(registry.Register(rms).ok());
    ASSERT_TRUE(registry.Register(linear).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto rms_kernels = registry.FindByOpType(OpType::kRmsNorm);
    ASSERT_TRUE(rms_kernels.ok());
    ASSERT_EQ(rms_kernels->size(), 1u);
    EXPECT_EQ((*rms_kernels)[0]->op_type, OpType::kRmsNorm);

    const auto linear_kernels = registry.FindByOpType(OpType::kLinear);
    ASSERT_TRUE(linear_kernels.ok());
    ASSERT_EQ(linear_kernels->size(), 1u);
    EXPECT_EQ((*linear_kernels)[0]->op_type, OpType::kLinear);

    const auto none = registry.FindByOpType(OpType::kSoftmax);
    ASSERT_TRUE(none.ok());
    EXPECT_TRUE(none->empty());
}

TEST(KernelRegistry, DebugDumpContainsRegisteredEntries) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(MakeTestKernelDef()).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const std::string dump = registry.DebugDump();
    EXPECT_FALSE(dump.empty());
    EXPECT_NE(dump.find("RmsNorm"), std::string::npos);
    EXPECT_NE(dump.find("test::op"), std::string::npos);
}

TEST(KernelRegistry, RegisterBorrowsKernelNameStorage) {
    KernelRegistry registry;
    static constexpr std::string_view kRegisteredName = "test::borrowed";
    KernelDef descriptor = MakeTestKernelDef();
    descriptor.name = kRegisteredName;

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const auto candidates = registry.FindCandidates(OpType::kRmsNorm, descriptor.selector);
    ASSERT_TRUE(candidates.ok()) << candidates.status().ToString();
    ASSERT_EQ(candidates->size(), 1U);
    // Registration borrows the name view; it must not rebind it to copied
    // storage, which is why callers must pass static-duration names.
    EXPECT_TRUE((*candidates)[0]->name.data() == kRegisteredName.data());
    EXPECT_EQ((*candidates)[0]->name, kRegisteredName);
}

TEST(KernelRegistry, ConcurrentFreezeIsIdempotent) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(MakeTestKernelDef()).ok());

    std::vector<std::thread> threads;
    std::vector<Status> statuses(8);
    threads.reserve(8);
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([&registry, &statuses, i] {
            statuses[i] = registry.Freeze();
        });
    }

    for (std::thread& thread: threads) {
        thread.join();
    }

    for (const Status& status: statuses) {
        EXPECT_TRUE(status.ok()) << status.ToString();
    }
    EXPECT_TRUE(registry.frozen());
    const auto candidates = registry.FindCandidates(
            OpType::kRmsNorm, MakeTestKernelDef().selector);
    ASSERT_TRUE(candidates.ok()) << candidates.status().ToString();
    ASSERT_EQ(candidates->size(), 1U);
    EXPECT_EQ((*candidates)[0]->kernel_func, &FakeKernel);
}

} // namespace
