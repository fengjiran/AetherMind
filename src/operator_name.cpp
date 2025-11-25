//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"
#include <sstream>

namespace aethermind {

std::optional<String> OperatorName::GetNamespace() const {
    auto pos = name_.find("::");
    if (pos == String::npos) {
        return std::nullopt;
    }
    return name_.substr(0, pos);
}

bool OperatorName::SetNamespaceIfNotSet(const char* ns) {
    if (GetNamespace().has_value()) {
        return false;
    }

    name_.insert(0, ns + String("::"));
    return true;
}


std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
    os << opName.name();
    if (!opName.overload_name().empty()) {
        os << "." << opName.overload_name();
    }
    return os;
}

String ToString(const OperatorName& opName) {
    std::ostringstream oss;
    oss << opName;
    return oss.str();
}

}// namespace aethermind
