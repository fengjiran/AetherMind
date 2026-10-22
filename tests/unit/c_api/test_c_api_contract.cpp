#include "c_api.h"

#include <gtest/gtest.h>

#include <cstddef>
#include <type_traits>

extern "C" int AetherMindCErrorContract(void);
extern "C" size_t AetherMindCConfigLayout(unsigned int field);

namespace {

static_assert(std::is_standard_layout_v<am_config_t>);
static_assert(std::is_standard_layout_v<am_session_config_t>);
static_assert(!std::is_convertible_v<am_runtime_t, am_model_t>);
static_assert(!std::is_convertible_v<am_model_t, am_session_t>);
static_assert(!std::is_convertible_v<am_session_t, am_runtime_t>);
static_assert(std::is_same_v<decltype(&am_get_abi_version), uint32_t (*)(void)>);
static_assert(std::is_same_v<decltype(&am_runtime_create), am_runtime_t (*)(const am_config_t*, am_error_handle*)>);
static_assert(std::is_same_v<decltype(&am_model_load), am_model_t (*)(am_runtime_t, const char*, am_error_handle*)>);
static_assert(std::is_same_v<decltype(&am_session_create), am_session_t (*)(am_model_t, const am_session_config_t*, am_error_handle*)>);
static_assert(std::is_same_v<decltype(&am_session_generate),
                             am_status_code (*)(am_session_t, const uint32_t*, size_t,
                                                uint32_t*, size_t, size_t*, am_error_handle*)>);

TEST(CAbiContract, CTranslationUnitCallsExistingErrorAbi) {
    EXPECT_EQ(AetherMindCErrorContract(), 0);
}

TEST(CAbiContract, RuntimeConfigLayoutMatchesC) {
    EXPECT_EQ(AetherMindCConfigLayout(0), sizeof(am_config_t));
    EXPECT_EQ(AetherMindCConfigLayout(1), offsetof(am_config_t, cpu_preset));
}

TEST(CAbiContract, SessionConfigLayoutMatchesC) {
    EXPECT_EQ(AetherMindCConfigLayout(2), sizeof(am_session_config_t));
    EXPECT_EQ(AetherMindCConfigLayout(3), offsetof(am_session_config_t, eos_token_id));
}

} // namespace
