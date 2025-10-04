//
// Created by richard on 10/4/25.
//

#include "scalar.h"

#include <fmt/format.h>
#include <fmt/ostream.h>

namespace aethermind {
std::ostream& operator<<(std::ostream& out, const Scalar& s) {
    if (s.isFloatingPoint()) {
        return out << s.toDouble();
    }

    if (s.isBool()) {
        return out << (s.toBool() ? "true" : "false");
    }

    if (s.isIntegral()) {
        return out << s.toLong();
    }

    // throw std::logic_error("Unknown type in Scalar");
    AETHERMIND_THROW(RuntimeError) << "Unknown type in Scalar";
    AETHERMIND_UNREACHABLE();
}

std::string toString(const Scalar& s) {
    return fmt::format("{}", fmt::streamed(s));
}
}// namespace aethermind