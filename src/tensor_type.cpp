//
// Created by richard on 10/2/25.
//
#include "tensor.h"
#include "type.h"

namespace aethermind {

std::atomic<size_t> ShapeSymbol::num_symbols_ = 1;

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s) {
    if (s.is_static()) {
        os << s.value();
    } else {
        os << "SS(" << s.value() << ')';
    }
    return os;
}

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        std::vector<ShapeSymbol> shape_symbols(rank.value());
        for (size_t i = 0; i < rank.value(); ++i) {
            shape_symbols[i] = ShapeSymbol::Create();
        }
        dims_ = shape_symbols;
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& dims) {
    std::vector<ShapeSymbol> shape_symbols(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i].has_value()) {
            shape_symbols[i] = ShapeSymbol::CreateFromStaticSize(dims[i].value());
        } else {
            shape_symbols[i] = ShapeSymbol::Create();
        }
    }
    dims_ = shape_symbols;
}


}// namespace aethermind