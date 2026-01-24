//
// Created by richard on 10/12/25.
//

#ifndef AETHERMIND_LAYOUT_H
#define AETHERMIND_LAYOUT_H

#include <cstdint>
#include <ostream>

namespace aethermind {

enum class Layout : uint8_t {
    Strided,
    Sparse,
    SparseCsr,
    Mkldnn,
    SparseCsc,
    SparseBsr,
    SparseBsc,
    Jagged,
    NumOptions
};

constexpr auto kStrided = Layout::Strided;
constexpr auto kSparse = Layout::Sparse;
constexpr auto kSparseCsr = Layout::SparseCsr;
constexpr auto kMkldnn = Layout::Mkldnn;
constexpr auto kSparseCsc = Layout::SparseCsc;
constexpr auto kSparseBsr = Layout::SparseBsr;
constexpr auto kSparseBsc = Layout::SparseBsc;
constexpr auto kJagged = Layout::Jagged;

inline std::ostream& operator<<(std::ostream& os, Layout layout) {
    switch (layout) {
        case kStrided:
            os << "Strided";
        case kSparse:
            os << "Sparse";
        case kSparseCsr:
            os << "SparseCsr";
        case kSparseCsc:
            os << "SparseCsc";
        case kSparseBsr:
            os << "SparseBsr";
        case kSparseBsc:
            os << "SparseBsc";
        case kMkldnn:
            os << "Mkldnn";
        case kJagged:
            os << "Jagged";
        default:
            AM_CHECK(false, "Unknown layout");
    }
    return os;
}

}// namespace aethermind

#endif//AETHERMIND_LAYOUT_H
