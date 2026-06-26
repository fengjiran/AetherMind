#include "aethermind/operators/operator_name.h"

#include <sstream>

namespace aethermind {

std::optional<std::string_view> OperatorName::GetNamespace() const noexcept {
    auto pos = name_.find("::");
    if (pos == std::string::npos) {
        return std::nullopt;
    }
    return std::string_view(name_).substr(0, pos);
}

std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
    os << opName.name();
    if (!opName.overload_name().empty()) {
        os << "." << opName.overload_name();
    }
    return os;
}

std::string ToString(const OperatorName& opName) {
    std::string result(opName.name());
    if (!opName.overload_name().empty()) {
        result += ".";
        result += opName.overload_name();
    }
    return result;
}

}// namespace aethermind
