#ifndef AETHERMIND_TEST_C_API_RUNTIME_HELPERS_H
#define AETHERMIND_TEST_C_API_RUNTIME_HELPERS_H

#include "c_api.h"

#include "aethermind/model/formats/hf/hf_model_config.h"

#include <filesystem>
#include <memory>
#include <type_traits>

namespace aethermind::test::c_api {

struct RuntimeDeleter {
    void operator()(am_runtime_t value) const noexcept { am_runtime_destroy(value); }
};
struct ModelDeleter {
    void operator()(am_model_t value) const noexcept { am_model_unload(value); }
};
struct SessionDeleter {
    void operator()(am_session_t value) const noexcept { am_session_destroy(value); }
};

using RuntimeOwner = std::unique_ptr<std::remove_pointer_t<am_runtime_t>, RuntimeDeleter>;
using ModelOwner = std::unique_ptr<std::remove_pointer_t<am_model_t>, ModelDeleter>;
using SessionOwner = std::unique_ptr<std::remove_pointer_t<am_session_t>, SessionDeleter>;

struct Error {
    am_error_handle value = nullptr;

    Error() = default;
    Error(const Error&) = delete;
    Error& operator=(const Error&) = delete;
    ~Error() { am_error_destroy(value); }

    void Clear() noexcept {
        am_error_destroy(value);
        value = nullptr;
    }

    const char* message() const noexcept { return am_error_message(value); }
};

inline std::filesystem::path Fixture(const char* name = "f32-explicit") {
    return std::filesystem::path(AETHERMIND_TEST_MODELS_DIR) / "inference-admission" / name;
}

inline am_config_t RuntimeConfig(size_t capacity = 8) {
    return {.struct_size = sizeof(am_config_t),
            .abi_version = AM_ABI_VERSION,
            .num_layers = 2,
            .num_kv_heads = 4,
            .head_dim = 4,
            .max_tokens = capacity,
            .kv_dtype = AM_KV_DTYPE_FLOAT32,
            .cpu_preset = AM_CPU_PRESET_O1_PLAIN};
}

inline am_config_t RuntimeConfig(const HfModelConfig& config, size_t capacity = 8) {
    auto result = RuntimeConfig(capacity);
    result.num_layers = static_cast<uint64_t>(config.num_hidden_layers);
    result.num_kv_heads = static_cast<uint64_t>(config.num_key_value_heads);
    result.head_dim = static_cast<uint64_t>(config.head_dim > 0
                                                    ? config.head_dim
                                                    : config.hidden_size / config.num_attention_heads);
    return result;
}

inline am_session_config_t SessionConfig(uint64_t count = 3) {
    return {.struct_size = sizeof(am_session_config_t),
            .abi_version = AM_ABI_VERSION,
            .max_new_tokens = count,
            .has_eos_token_id = 0,
            .eos_token_id = 0};
}

} // namespace aethermind::test::c_api

#endif
