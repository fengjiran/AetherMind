//
// Created by richard on 10/4/25.
//

#include "scalar.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace aethermind {
std::ostream& operator<<(std::ostream& out, const Scalar& s) {
    if (s.is_floating_point()) {
        return out << s.toDouble();
    }

    if (s.is_bool()) {
        return out << (s.toBool() ? "true" : "false");
    }

    if (s.is_integral()) {
        return out << s.toLong();
    }

    AETHERMIND_THROW(RuntimeError) << "Unknown type in Scalar";
    AETHERMIND_UNREACHABLE();
}

std::string toString(const Scalar& s) {
    return fmt::format("{}", fmt::streamed(s));
}
}// namespace aethermind