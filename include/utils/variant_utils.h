#ifndef AETHERMIND_UTILS_VARIANT_UTILS_H
#define AETHERMIND_UTILS_VARIANT_UTILS_H

namespace aethermind {

template<typename... Ts>
// NOLINTNEXTLINE(readability-identifier-naming)
struct overloaded : Ts... {
    using Ts::operator()...;
};

// CTAD deduction guide
template<typename... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}// namespace aethermind

#endif// AETHERMIND_UTILS_VARIANT_UTILS_H
