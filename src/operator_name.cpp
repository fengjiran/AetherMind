//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"

#include <sstream>

namespace aethermind {

std::optional<std::string_view> OperatorName::GetNamespace() const noexcept {
    auto pos = name_.find("::");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return std::string_view(name_).substr(0, pos);
}

bool OperatorName::SetNamespaceIfNotSet(std::string_view ns) {
    if (GetNamespace().has_value()) {
        return false;
    }
    name_.insert(0, std::string(ns) + "::");
    return true;
}

std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
    os << opName.name();
    if (!opName.overload_name().empty()) {
        os << "." << opName.overload_name();
    }
    return os;
}

std::string ToString(const OperatorName& opName) {
    std::ostringstream oss;
    oss << opName;
    return oss.str();
}

}// namespace aethermind