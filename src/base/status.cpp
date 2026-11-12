#include "aethermind/base/status.h"

#include <ostream>

namespace aethermind {

std::ostream& operator<<(std::ostream& os, const Status& status) {
    return os << status.ToString();
}

} // namespace aethermind
