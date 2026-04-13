#include <gtest/gtest.h>

namespace {

TEST(BackendRegistry, RegisterFactoryStoresFactory) {
    GTEST_SKIP() << "Phase 1 skeleton: add BackendRegistry/RegisterFactory assertions after backend interfaces land.";
    // TODO:
    // 1. Construct BackendRegistry.
    // 2. Register a fake CPU BackendFactory.
    // 3. Verify the first GetBackend(DeviceType::kCPU) succeeds.
}

TEST(BackendRegistry, GetBackendLazyCreatesInstance) {
    GTEST_SKIP() << "Phase 1 skeleton: verify lazy backend instantiation once BackendRegistry is implemented.";
    // TODO:
    // 1. Register a counting fake factory.
    // 2. Assert create_count == 0 before first GetBackend().
    // 3. Assert create_count == 1 after first GetBackend().
}

TEST(BackendRegistry, GetBackendCachesInstance) {
    GTEST_SKIP() << "Phase 1 skeleton: verify backend instance caching once BackendRegistry is implemented.";
    // TODO:
    // 1. Register a counting fake factory.
    // 2. Call GetBackend(DeviceType::kCPU) twice.
    // 3. Assert the returned backend identity is stable.
    // 4. Assert create_count == 1.
}

TEST(BackendRegistry, GetBackendForUnregisteredDeviceFails) {
    GTEST_SKIP() << "Phase 1 skeleton: define failure-path assertions after GetBackend error contract is finalized.";
    // TODO:
    // 1. Construct an empty BackendRegistry.
    // 2. Call GetBackend() for an unregistered device.
    // 3. Assert the agreed failure behavior (StatusOr or exception).
}

TEST(BackendRegistry, OverrideFactoryBeforeInstantiationUsesLatestFactory) {
    GTEST_SKIP() << "Phase 1 skeleton: verify pre-instantiation factory override semantics after BackendRegistry is implemented.";
    // TODO:
    // 1. Register fake factory A for CPU.
    // 2. Register fake factory B for CPU before first GetBackend().
    // 3. Assert the created backend comes from factory B.
}

TEST(BackendRegistry, OverrideFactoryAfterInstantiationKeepsCachedInstance) {
    GTEST_SKIP() << "Phase 1 skeleton: verify cached-instance semantics after BackendRegistry is implemented.";
    // TODO:
    // 1. Register fake factory A and instantiate backend.
    // 2. Register fake factory B for the same DeviceType.
    // 3. Assert subsequent GetBackend() returns the original cached instance.
    // 4. Assert factory B was not used to replace the live backend.
}

}// namespace
