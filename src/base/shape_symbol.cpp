#include "aethermind/base/shape_symbol.h"

namespace aethermind {

std::atomic<int64_t> ShapeSymbol::next_symbol_ = -2;

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    if (s.IsStatic()) {
        return os << s.GetStaticValue();
    }

    if (s.IsUnknown()) {
        return os << "?";
    }

    return os << "S" << -s.value();

    // if (s.IsStatic()) {
    //     os << s.value();
    // } else {
    //     os << "SS(" << s.value() << ')';
    // }
    // return os;
}

}// namespace aethermind
