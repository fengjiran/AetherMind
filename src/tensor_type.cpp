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

SymbolicShape::SymbolicShape(IntArrayView dims) {
    std::vector<ShapeSymbol> shape_symbols(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        shape_symbols[i] = ShapeSymbol::CreateFromStaticSize(dims[i]);
    }
    dims_ = shape_symbols;
}

ShapeSymbol SymbolicShape::operator[](size_t i) const {
    if (!dims_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }
    return dims_.value()[i];
}

ShapeSymbol SymbolicShape::at(size_t i) const {
    if (!dims_.has_value()) {
        AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
    }

    if (i >= dims_->size()) {
        AETHERMIND_THROW(OutOfRangeError) << "Out of range";
    }
    return dims_.value()[i];
}

std::optional<size_t> SymbolicShape::rank() const {
    if (dims_.has_value()) {
        return dims_->size();
    }
    return std::nullopt;
}

std::optional<std::vector<ShapeSymbol>> SymbolicShape::sizes() const {
    return dims_;
}

std::optional<std::vector<bool>> SymbolicShape::symbolic_dims() const {
    if (!dims_.has_value()) {
        return std::nullopt;
    }

    std::vector<bool> res(rank().value());
    for (size_t i = 0; i < rank().value(); ++i) {
        res[i] = !dims_.value()[i].is_static();
    }
    return res;
}

bool SymbolicShape::is_complete() const {
    if (!dims_.has_value()) {
        return false;
    }

    for (auto d: dims_.value()) {
        if (!d.is_static()) {
            return false;
        }
    }
    return true;
}

SymbolicShape SymbolicShape::merge(const SymbolicShape& other) const {
    if (!dims_.has_value() || !other.dims_.has_value() || dims_->size() != other.dims_->size()) {
        return {};
    }

    auto n = dims_->size();
    std::vector<ShapeSymbol> dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = merge_primitive(dims_.value()[i], other.dims_.value()[i]);
    }
    return {std::move(dims)};
}


}// namespace aethermind