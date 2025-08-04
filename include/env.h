//
// Created by richard on 6/25/25.
//

#ifndef ENV_H
#define ENV_H

#include <vector>
#include <string>
#include <optional>

namespace atp {

void set_env(const char* name, const char* value, bool overwrite);

std::optional<std::string> get_env(const char* name) noexcept;

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

    std::vector<std::string> names_;
};

}

#endif //ENV_H
