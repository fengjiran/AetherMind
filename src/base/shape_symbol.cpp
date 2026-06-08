#include "aethermind/base/shape_symbol.h"

namespace aethermind {

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    if (s.IsStatic()) {
        os << s.value();
    } else {
        os << "SS(" << s.value() << ')';
    }
    return os;
}

}// namespace aethermind
