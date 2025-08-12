//
// Created by 赵丹 on 2025/8/12.
//

#include "operator_name.h"
#include <sstream>

namespace aethermind {

std::ostream& operator<<(std::ostream& os, const OperatorName& opName) {
    os << opName.name();
    if (!opName.overload_name().empty()) {
        os << "." << opName.overload_name();
    }
    return os;
}

std::string toString(const OperatorName& opName) {
    std::ostringstream oss;
    oss << opName;
    return oss.str();
}

}
