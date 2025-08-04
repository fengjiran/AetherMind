//
// Created by richard on 6/25/25.
//

#include "env.h"

#include <cstdlib>
#include <fmt/format.h>
#include <glog/logging.h>
#include <mutex>
#include <shared_mutex>

namespace atp {

static std::shared_mutex& get_env_mutex() {
    static std::shared_mutex env_mutex;
    return env_mutex;
}

void set_env(const char* name, const char* value, bool overwrite) {
    std::lock_guard lock(get_env_mutex());
    auto err = setenv(name, value, overwrite);
    CHECK(err == 0);
}

std::optional<std::string> get_env(const char* name) noexcept {
    std::shared_lock lock(get_env_mutex());
    auto env_value = std::getenv(name);
    if (env_value != nullptr) {
        return std::string(env_value);
    }
    return std::nullopt;
}

bool has_env(const char* name) noexcept {
    return get_env(name).has_value();
}

std::optional<bool> check_env(const char* name) {
    auto env_opt = get_env(name);
    if (env_opt.has_value()) {
        if (env_opt == "0") {
            return false;
        }

        if (env_opt == "1") {
            return true;
        }
    }
    return std::nullopt;
}

static auto& env = RegisterEnvs::Global().SetEnv("THP_MEM_ALLOC_ENABLE", "1", false);

}// namespace atp
