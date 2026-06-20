#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/kernel_registry.h"
#include "aethermind/runtime/workspace.h"

#include "aethermind/dtypes/data_type.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <thread>
#include <vector>

using namespace aethermind;

namespace {

Status FakeKernel(const KernelContext&) noexcept {
    return Status::Ok();
}

KernelDescriptor MakeTestKernelDescriptor() {
    return KernelDescriptor{
            .op_type = OpType::kRmsNorm,
            .selector = KernelSelector{
                    .device_type = DeviceType::kCPU,
                    .activation_dtype = DataType::Float32(),
                    .weight_dtype = DataType::Float32(),
                    .weight_format = WeightFormat::kPlain,
                    .isa = IsaLevel::kScalar,
                    .phase = ExecPhase::kBoth,
            },
            .kernel_func = &FakeKernel,
            .name = "test::op",
            .priority = 1,
    };
}

KernelSelector MakeMissingSelector() {
    return KernelSelector{
            .device_type = DeviceType::kCPU,
            .activation_dtype = DataType::Float32(),
            .weight_dtype = DataType::Float32(),
            .weight_format = WeightFormat::kPlain,
            .isa = IsaLevel::kScalar,
            .phase = ExecPhase::kBoth,
    };
}

TEST(KernelRegistry, RegisterAndLookupReturnsKernel) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);
    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, RequestBasedResolveReturnsKernel) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);

    ASSERT_TRUE(resolved.ok());
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

TEST(KernelRegistry, LookupMissingKeyReturnsNullptr) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kLinear, MakeMissingSelector());
    EXPECT_EQ(resolved.status().code(), StatusCode::kNotFound);
}

TEST(KernelRegistry, RequestRejectsUnknownOpType) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kUnknown, MakeMissingSelector());

    ASSERT_FALSE(resolved.ok());
    EXPECT_EQ(resolved.status().code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, ResolveBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);
    EXPECT_EQ(resolved.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, FindByOpTypeBeforeFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());

    const auto found = registry.FindByOpType(OpType::kRmsNorm);
    EXPECT_EQ(found.status().code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, RegisterAfterFreezeFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    ASSERT_TRUE(registry.Freeze().ok());
    EXPECT_TRUE(registry.frozen());

    KernelDescriptor extra = descriptor;
    extra.op_type = OpType::kLinear;
    extra.name = "test::other_op";

    const Status status = registry.Register(extra);
    EXPECT_EQ(status.code(), StatusCode::kFailedPrecondition);
}

TEST(KernelRegistry, DuplicateRegistrationFails) {
    KernelRegistry registry;
    const KernelDescriptor descriptor = MakeTestKernelDescriptor();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    EXPECT_EQ(registry.Register(descriptor).code(), StatusCode::kAlreadyExists);
}

TEST(KernelRegistry, RegisterRejectsEmptyKernelName) {
    KernelRegistry registry;
    KernelDescriptor descriptor = MakeTestKernelDescriptor();
    descriptor.name.clear();

    const Status status = registry.Register(descriptor);

    EXPECT_EQ(status.code(), StatusCode::kInvalidArgument);
}

TEST(KernelRegistry, FindByOpTypeReturnsMatchingDescriptors) {
    KernelRegistry registry;
    const KernelDescriptor rms = MakeTestKernelDescriptor();

    KernelDescriptor linear = rms;
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
    ASSERT_TRUE(registry.Register(MakeTestKernelDescriptor()).ok());
    ASSERT_TRUE(registry.Freeze().ok());

    const std::string dump = registry.DebugDump();
    EXPECT_FALSE(dump.empty());
    EXPECT_NE(dump.find("RmsNorm"), std::string::npos);
    EXPECT_NE(dump.find("test::op"), std::string::npos);
}

TEST(KernelRegistry, RegisterCopiesKernelNameStorage) {
    KernelRegistry registry;
    KernelDescriptor descriptor = MakeTestKernelDescriptor();
    std::array<char, 15> mutable_name{};
    const char original_name[] = "test::mutable";
    std::copy_n(original_name, sizeof(original_name), mutable_name.data());
    descriptor.name = mutable_name.data();

    ASSERT_TRUE(registry.Register(descriptor).ok());
    mutable_name.fill('x');
    ASSERT_TRUE(registry.Freeze().ok());

    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, descriptor.selector);
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ((*resolved)->name, original_name);
}

TEST(KernelRegistry, ConcurrentFreezeIsIdempotent) {
    KernelRegistry registry;
    ASSERT_TRUE(registry.Register(MakeTestKernelDescriptor()).ok());

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
    const StatusOr<const KernelDescriptor*> resolved =
            registry.Resolve(OpType::kRmsNorm, MakeTestKernelDescriptor().selector);
    ASSERT_TRUE(resolved.ok()) << resolved.status().ToString();
    EXPECT_EQ((*resolved)->kernel_func, &FakeKernel);
}

}// namespace
