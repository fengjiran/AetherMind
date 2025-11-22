//
// Created by richard on 6/25/25.
//

#ifndef AETHERMIND_ENV_H
#define AETHERMIND_ENV_H

#include "container/string.h"

#include <vector>

namespace aethermind {

void set_env(const char* name, const char* value, bool overwrite);

std::optional<String> get_env(const char* name) noexcept;

bool has_env(const char* name) noexcept;

std::optional<bool> check_env(const char* name);

class RegisterEnvs {
public:
    static RegisterEnvs& Global() {
        static RegisterEnvs inst;
        return inst;
    }

    RegisterEnvs& SetEnv(const char* name, const char* value, bool overwrite) {
        set_env(name, value, overwrite);
        names_.emplace_back(name);
        return *this;
    }

private:
    RegisterEnvs() = default;

    std::vector<String> names_;
};

}// namespace aethermind

#endif//AETHERMIND_ENV_H
