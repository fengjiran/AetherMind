#include "aethermind/backend/backend.h"
#include "aethermind/backend/backend_factory.h"
#include "aethermind/backend/backend_registry.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/operators/op_type.h"

#include <gtest/gtest.h>

namespace {

using namespace aethermind;

class FakeBackend : public Backend {
public:
    explicit FakeBackend(DeviceType type) : type_(type) {
        caps_.device_type = type;
    }

    DeviceType device_type() const noexcept override { return type_; }
    const BackendCapabilities& capabilities() const noexcept override { return caps_; }
    KernelFunc ResolveKernel(OpType, const KernelSelector&) const noexcept override { return nullptr; }
    const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept override { return nullptr; }
    StatusOr<ResolvedKernel> ResolveKernelInfo(OpType op_type, const KernelSelector& selector) const noexcept override {
        const KernelFunc fn = ResolveKernel(op_type, selector);
        if (fn == nullptr) {
            return Status::NotFound(
                    "No matching kernel registered: op_type=" +
                    std::string(ToString(op_type)) +
                    ", selector=" + ToString(selector));
        }

        return ResolvedKernel{
                .op_type = op_type,
                .fn = fn,
                .attrs = {},
                .debug_name = nullptr,
        };
    }

private:
    DeviceType type_;
    BackendCapabilities caps_;
};

class FakeFactory : public BackendFactory {
public:
    explicit FakeFactory(DeviceType type, int* create_count = nullptr)
        : type_(type), create_count_(create_count) {}

    DeviceType device_type() const noexcept override { return type_; }
    std::unique_ptr<Backend> Create() const override {
        if (create_count_) {
            (*create_count_)++;
        }
        return std::make_unique<FakeBackend>(type_);
    }

private:
    DeviceType type_;
    int* create_count_;
};

TEST(BackendRegistry, RegisterFactoryStoresFactory) {
    BackendRegistry registry;
    registry.RegisterFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU));

    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(status_or_backend.ok());
    ASSERT_NE(status_or_backend.value(), nullptr);
    EXPECT_EQ(status_or_backend.value()->device_type(), DeviceType::kCPU);
}

TEST(BackendRegistry, GetBackendLazyCreatesInstance) {
    BackendRegistry registry;
    int create_count = 0;
    registry.RegisterFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &create_count));

    EXPECT_EQ(create_count, 0);

    auto status_or_backend = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(status_or_backend.ok());
    EXPECT_EQ(create_count, 1);
}

TEST(BackendRegistry, GetBackendCachesInstance) {
    BackendRegistry registry;
    int create_count = 0;
    registry.RegisterFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &create_count));

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

    registry.RegisterFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &count1));
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &count2));

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
    registry.RegisterFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &count1));
    auto res1 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res1.ok());
    Backend* b1 = res1.value();
    EXPECT_EQ(count1, 1);

    // 2. Override with factory B.
    // According to production code, SetFactory clears the cached backend for that DeviceType.
    registry.SetFactory(DeviceType::kCPU, std::make_unique<FakeFactory>(DeviceType::kCPU, &count2));

    // 3. Assert subsequent GetBackend() returns a NEW instance from factory B.
    auto res2 = registry.GetBackend(DeviceType::kCPU);
    ASSERT_TRUE(res2.ok());
    Backend* b2 = res2.value();

    EXPECT_NE(b1, b2);
    EXPECT_EQ(count2, 1);
}

}// namespace
