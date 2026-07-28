#ifndef AETHERMIND_OPERATORS_TEST_OPERATOR_SEMANTICS_HELPERS_H
#define AETHERMIND_OPERATORS_TEST_OPERATOR_SEMANTICS_HELPERS_H

#include "aethermind/shape_inference/tensor_spec.h"

#include <cstdint>
#include <vector>

namespace aethermind {

// Builds an unranked TensorSpec with the given dtype.
inline TensorSpec MakeSpec(DataType dtype) {
    return {dtype, SymbolicShape(std::nullopt)};
}

// Builds a fully-static TensorSpec from a dtype and concrete dims.
inline TensorSpec MakeSpec(DataType dtype, const std::vector<int64_t>& dims) {
    std::vector<ShapeSymbol> symbols;
    for (auto d: dims) {
        symbols.push_back(ShapeSymbol::CreateFromValue(d));
    }
    return {dtype, SymbolicShape(symbols)};
}

// Builds a TensorSpec with the given dtype, rank, and fresh symbolic dims.
inline TensorSpec MakeSymbolicSpec(DataType dtype, size_t rank) {
    std::vector<ShapeSymbol> symbols;
    symbols.reserve(rank);
    for (size_t i = 0; i < rank; ++i) {
        symbols.push_back(ShapeSymbol::Create());
    }
    return {dtype, SymbolicShape(symbols)};
}

}// namespace aethermind

#endif// AETHERMIND_OPERATORS_TEST_OPERATOR_SEMANTICS_HELPERS_H
