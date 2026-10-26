#include <cstdlib>
#include <cstring>
#include <gtest/gtest.h>

namespace aethermind::test {
namespace {

class RequiredExecutionGuard final : public ::testing::EmptyTestEventListener {
    void OnTestEnd(const ::testing::TestInfo& info) override {
        const char* const required = std::getenv("AETHERMIND_REQUIRED_VALIDATION");
        if (required != nullptr && std::strcmp(required, "1") == 0 && info.result()->Skipped()) {
            ADD_FAILURE_AT(info.file(), info.line()) << "Required validation case skipped";
        }
    }
};

class RequiredExecutionEnvironment final : public ::testing::Environment {
    void SetUp() override {
        // GTest owns the listener. Environment setup follows printer selection;
        // appending here runs this OnTestEnd before either printer's callback.
        [[maybe_unused]] static const bool installed = [] {
            ::testing::UnitTest::GetInstance()->listeners().Append(new RequiredExecutionGuard);
            return true;
        }();
    }
};

[[maybe_unused]] auto* const environment =
        ::testing::AddGlobalTestEnvironment(new RequiredExecutionEnvironment);

} // namespace

} // namespace aethermind::test
