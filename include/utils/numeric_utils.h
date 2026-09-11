#ifndef AETHERMIND_UTILS_NUMERIC_UTILS_H
#define AETHERMIND_UTILS_NUMERIC_UTILS_H

/// @file
/// @brief Small numeric predicates shared across layers.

#include <cmath>

namespace aethermind {

/// @brief Reports whether a value is finite and strictly positive.
inline bool IsFinitePositive(double value) noexcept {
    return std::isfinite(value) && value > 0.0;
}

} // namespace aethermind

#endif // AETHERMIND_UTILS_NUMERIC_UTILS_H
