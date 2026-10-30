#include "aethermind/runtime/runtime_builder.h"

#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "test_utils/test_resource_allocator.h"

#include <array>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>

namespace {

using namespace aethermind;

struct ResourceObservation {
    size_t allocator_creations = 0;
    size_t backend_creations = 0;
    size_t provider_destructions = 0;
    size_t factory_destructions = 0;
    std::shared_ptr<test::ResourceAllocationStats> storage = std::make_shared<test::ResourceAllocationStats>();
};

class ObservedProvider final : public AllocatorProvider {
public:
    explicit ObservedProvider(std::shared_ptr<ResourceObservation> observation)
        : observation_(std::move(observation)) {}

    ~ObservedProvider() override { ++observation_->provider_destructions; }

    std::unique_ptr<Allocator> CreateAllocator(Device device) override {
        EXPECT_EQ(device, Device::CPU());
        ++observation_->allocator_creations;
        return std::make_unique<test::ResourceRecordingAllocator>(observation_->storage);
    }

private:
    std::shared_ptr<ResourceObservation> observation_;
};

class ObservedBackend final : public Backend {
public:
    explicit ObservedBackend(DeviceType type) : type_(type) {}

    DeviceType device_type() const noexcept override { return type_; }

    StatusOr<ResolvedKernel> PrepareKernel(OpType, const KernelSelector&, const OpParams&) const override {
        return Status::Unimplemented("Test backend does not resolve kernels");
    }

    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }

private:
    DeviceType type_;
};

class ObservedFactory final : public BackendFactory {
public:
    ObservedFactory(DeviceType type, std::shared_ptr<ResourceObservation> observation)
        : type_(type), observation_(std::move(observation)) {}

    ~ObservedFactory() override { ++observation_->factory_destructions; }

    DeviceType device_type() const noexcept override { return type_; }

    StatusOr<std::unique_ptr<Backend>> Create() const override {
        ++observation_->backend_creations;
        return std::make_unique<ObservedBackend>(type_);
    }

private:
    DeviceType type_;
    std::shared_ptr<ResourceObservation> observation_;
};

RuntimeOptions KvOptions() {
    RuntimeOptions options;
    options.kv_cache = {.enable_manager = true,
                        .num_layers = 1,
                        .num_kv_heads = 1,
                        .max_tokens = 2,
                        .head_dim = 4,
                        .kv_dtype = DataType::Float32(),
                        .alignment = 64};
    return options;
}

void ObserveCpu(RuntimeBuilder& builder, const std::shared_ptr<ResourceObservation>& observation) {
    builder.RegisterCustomAllocatorProvider(DeviceType::kCPU, std::make_unique<ObservedProvider>(observation));
    builder.RegisterBackendFactory(DeviceType::kCPU, std::make_unique<ObservedFactory>(DeviceType::kCPU, observation));
}

void ExpectNoResourceUse(const ResourceObservation& observation) {
    EXPECT_EQ(observation.allocator_creations, 0U);
    EXPECT_EQ(observation.backend_creations, 0U);
    EXPECT_EQ(observation.provider_destructions, 0U);
    EXPECT_EQ(observation.factory_destructions, 0U);
    EXPECT_TRUE(observation.storage->requests.empty());
}

struct UnsupportedOption {
    const char* field;
    void (*set)(RuntimeOptions&);
};

TEST(RuntimeOptionsValidation, RejectsUnsupportedOptionsBeforeCreatingResourcesAndCanRetry) {
    const std::array cases{
            UnsupportedOption{"execution.enable_default_stream", [](RuntimeOptions& options) { options.execution.enable_default_stream = false; }},
            UnsupportedOption{"execution.enable_graph_executor", [](RuntimeOptions& options) { options.execution.enable_graph_executor = true; }},
            UnsupportedOption{"workspace.enable_workspace_manager", [](RuntimeOptions& options) { options.workspace.enable_workspace_manager = true; }},
            UnsupportedOption{"tracing.enable_profiling", [](RuntimeOptions& options) { options.tracing.enable_profiling = true; }},
            UnsupportedOption{"tracing.enable_memory_profiling", [](RuntimeOptions& options) { options.tracing.enable_memory_profiling = true; }}};

    for (const auto& option: cases) {
        SCOPED_TRACE(option.field);
        auto options = KvOptions();
        option.set(options);
        RuntimeBuilder builder;
        builder.WithOptions(options);
        auto observation = std::make_shared<ResourceObservation>();
        ObserveCpu(builder, observation);

        const auto rejected = builder.TryBuild();

        EXPECT_EQ(rejected.status().code(), StatusCode::kUnimplemented);
        EXPECT_NE(rejected.status().message().find(option.field), std::string::npos);
        ExpectNoResourceUse(*observation);

        builder.WithOptions(KvOptions());
        auto retry = builder.TryBuild();
        ASSERT_TRUE(retry.ok()) << retry.status().ToString();
        ASSERT_NE(retry->GetKVCacheManager(), nullptr);
        EXPECT_EQ(observation->allocator_creations, 1U);
        EXPECT_EQ(observation->storage->requests.size(), 2U);
        EXPECT_EQ(observation->backend_creations, 0U);
        const auto backend = retry->GetBackend(DeviceType::kCPU);
        ASSERT_TRUE(backend.ok()) << backend.status().ToString();
        EXPECT_EQ(observation->backend_creations, 1U);
    }
}

TEST(RuntimeOptionsValidation, RejectsGpuDefaultsWithoutMatchingCustomFactories) {
    for (const auto type: {DeviceType::kCUDA, DeviceType::kCANN}) {
        auto options = KvOptions();
        const bool cuda = type == DeviceType::kCUDA;
        (cuda ? options.backend.enable_cuda : options.backend.enable_cann) = true;
        RuntimeBuilder builder;
        builder.WithOptions(options);
        auto observation = std::make_shared<ResourceObservation>();
        ObserveCpu(builder, observation);

        const auto rejected = builder.TryBuild();

        EXPECT_EQ(rejected.status().code(), StatusCode::kUnimplemented);
        EXPECT_NE(rejected.status().message().find(cuda ? "backend.enable_cuda" : "backend.enable_cann"), std::string::npos);
        ExpectNoResourceUse(*observation);

        builder.WithOptions(KvOptions());
        auto retry = builder.TryBuild();
        ASSERT_TRUE(retry.ok()) << retry.status().ToString();
        EXPECT_EQ(observation->allocator_creations, 1U);
        EXPECT_EQ(observation->storage->requests.size(), 2U);
        ASSERT_TRUE(retry->GetBackend(DeviceType::kCPU).ok());
        EXPECT_EQ(observation->backend_creations, 1U);
    }
}

TEST(RuntimeOptionsValidation, AcceptsGpuFlagsWithMatchingCustomFactories) {
    for (const auto type: {DeviceType::kCUDA, DeviceType::kCANN}) {
        RuntimeOptions options;
        options.allocator.enable_cpu = false;
        options.backend.enable_cpu = false;
        (type == DeviceType::kCUDA ? options.backend.enable_cuda : options.backend.enable_cann) = true;
        RuntimeBuilder builder;
        builder.WithOptions(options);
        auto observation = std::make_shared<ResourceObservation>();
        builder.RegisterBackendFactory(type, std::make_unique<ObservedFactory>(type, observation));

        auto runtime = builder.TryBuild();

        ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
        EXPECT_EQ(observation->backend_creations, 0U);
        const auto backend = runtime->GetBackend(type);
        ASSERT_TRUE(backend.ok()) << backend.status().ToString();
        EXPECT_EQ((*backend)->device_type(), type);
        EXPECT_EQ(observation->backend_creations, 1U);
    }
}

TEST(RuntimeOptionsValidation, GpuRejectionPreservesMatchedFactoriesForRetry) {
    RuntimeOptions options;
    options.allocator.enable_cpu = false;
    options.backend.enable_cpu = false;
    options.backend.enable_cuda = true;
    options.backend.enable_cann = true;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    auto cuda = std::make_shared<ResourceObservation>();
    builder.RegisterBackendFactory(DeviceType::kCUDA, std::make_unique<ObservedFactory>(DeviceType::kCUDA, cuda));

    const auto rejected = builder.TryBuild();

    EXPECT_EQ(rejected.status().code(), StatusCode::kUnimplemented);
    EXPECT_NE(rejected.status().message().find("backend.enable_cann"), std::string::npos);
    ExpectNoResourceUse(*cuda);
    auto cann = std::make_shared<ResourceObservation>();
    builder.RegisterBackendFactory(DeviceType::kCANN, std::make_unique<ObservedFactory>(DeviceType::kCANN, cann));
    auto retry = builder.TryBuild();
    ASSERT_TRUE(retry.ok()) << retry.status().ToString();
    ASSERT_TRUE(retry->GetBackend(DeviceType::kCUDA).ok());
    ASSERT_TRUE(retry->GetBackend(DeviceType::kCANN).ok());
    EXPECT_EQ(cuda->backend_creations, 1U);
    EXPECT_EQ(cann->backend_creations, 1U);
}

TEST(RuntimeOptionsValidation, CustomCpuAllocatorCanSupplyKvWhenDefaultIsDisabled) {
    auto options = KvOptions();
    options.allocator.enable_cpu = false;
    options.backend.enable_cpu = false;
    RuntimeBuilder builder;
    builder.WithOptions(options);
    auto observation = std::make_shared<ResourceObservation>();
    builder.RegisterCustomAllocatorProvider(DeviceType::kCPU, std::make_unique<ObservedProvider>(observation));

    const auto runtime = builder.TryBuild();

    ASSERT_TRUE(runtime.ok()) << runtime.status().ToString();
    EXPECT_NE(runtime->GetKVCacheManager(), nullptr);
    EXPECT_EQ(observation->allocator_creations, 1U);
    EXPECT_EQ(observation->storage->requests.size(), 2U);
}

} // namespace
