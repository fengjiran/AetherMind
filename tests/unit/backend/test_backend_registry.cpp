#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/backend_registry.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

class FakeBackend : public Backend {
public:
    explicit FakeBackend(DeviceType type) : type_(type) {}

    DeviceType device_type() const noexcept override { return type_; }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }
    StatusOr<ResolvedKernel> PrepareKernel(OpType,
                                           const KernelSelector&,
                                           const OpParams&) const override {
        return Status::NotFound("FakeBackend has no kernels");
    }

private:
    DeviceType type_;
};

class FakeBackendFactory : public BackendFactory {
public:
    explicit FakeBackendFactory(DeviceType type, int* create_count = nullptr)
        : type_(type), create_count_(create_count) {}

    DeviceType device_type() const noexcept override { return type_; }
    StatusOr<std::unique_ptr<Backend>> Create() const override {
        if (create_count_) {
            (*create_count_)++;
        }
        return std::make_unique<FakeBackend>(type_);
    }

private:
    DeviceType type_;
    int* create_count_;
};

TEST(BackendRegistry, SetFactoryStoresFactory) {
    BackendRegistry registry;
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU));

    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(status_or_backend.ok());
    ASSERT_NE(status_or_backend.value(), nullptr);
    EXPECT_EQ(status_or_backend.value()->device_type(), DeviceType::kCPU);
}

TEST(BackendRegistry, GetBackendLazyCreatesInstance) {
    BackendRegistry registry;
    int create_count = 0;
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &create_count));

    EXPECT_EQ(create_count, 0);

    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(status_or_backend.ok());
    EXPECT_EQ(create_count, 1);
}

TEST(BackendRegistry, GetBackendCachesInstance) {
    BackendRegistry registry;
    int create_count = 0;
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &create_count));

    auto res1 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res1.ok());
    Backend* b1 = res1.value();

    auto res2 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res2.ok());
    Backend* b2 = res2.value();

    EXPECT_EQ(b1, b2);
    EXPECT_EQ(create_count, 1);
}

TEST(BackendRegistry, GetBackendForUnregisteredDeviceFails) {
    BackendRegistry registry;
    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    EXPECT_FALSE(status_or_backend.ok());
    EXPECT_EQ(status_or_backend.status().code(), StatusCode::kNotFound);
}

TEST(BackendRegistry, OverrideFactoryBeforeInstantiationUsesLatestFactory) {
    BackendRegistry registry;
    int count1 = 0;
    int count2 = 0;

    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &count1));
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &count2));

    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(status_or_backend.ok());
    EXPECT_EQ(count1, 0);
    EXPECT_EQ(count2, 1);
}

TEST(BackendRegistry, OverrideFactoryAfterInstantiationClearsCachedInstance) {
    BackendRegistry registry;
    int count1 = 0;
    int count2 = 0;

    // 1. Register fake factory A and instantiate backend.
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &count1));
    auto res1 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res1.ok());
    EXPECT_EQ(count1, 1);

    // 2. Override with factory B. SetFactory must clear the cached backend
    // so the next GetBackend() re-creates from the new factory.
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeBackendFactory>(DeviceType::kCPU, &count2));

    // 3. Factory B is invoked only if the cached instance was cleared, so
    // count2 == 1 is the behavioral proof. Pointer inequality is not a valid
    // check here: the freed instance's address is typically reused by the
    // allocator, so equal raw pointers do not mean the instance was cached.
    auto res2 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res2.ok());
    EXPECT_EQ(count2, 1);
}

} // namespace
