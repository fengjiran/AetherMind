#include "aethermind/base/shape_symbol.h"
#include "error.h"

#include <algorithm>

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
}


std::ostream& operator<<(std::ostream& os, const SymbolicShape& s) {
    const auto rank_opt = s.rank();
    if (!rank_opt.has_value()) {
        os << "(*)";
        return os;
    }

    const auto& shape = s.shape().value();
    os << "(";
    for (size_t i = 0; i < rank_opt.value(); ++i) {
        if (i > 0) {
            os << ", ";
        }

        if (shape[i].IsStatic()) {
            os << shape[i];
        } else {
            os << "*";
        }
    }
    os << ")";
    return os;
}

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        std::vector<ShapeSymbol> symbolic_shape(rank.value());
        for (size_t i = 0; i < rank.value(); ++i) {
            symbolic_shape[i] = ShapeSymbol::Create();
        }
        symbolic_shape_ = symbolic_shape;
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& shape) {
    std::vector<ShapeSymbol> symbolic_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i].has_value()) {
            symbolic_shape[i] = ShapeSymbol::CreateFromValue(shape[i].value());
        } else {
            symbolic_shape[i] = ShapeSymbol::Create();
        }
    }
    symbolic_shape_ = symbolic_shape;
}

SymbolicShape::SymbolicShape(IntArrayView shape) {
    std::vector<ShapeSymbol> symbolic_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        symbolic_shape[i] = ShapeSymbol::CreateFromValue(shape[i]);
    }
    symbolic_shape_ = symbolic_shape;
}

ShapeSymbol SymbolicShape::operator[](size_t i) const {
    if (!symbolic_shape_.has_value()) {
        AM_THROW(RuntimeError) << "Rank isn't fixed";
    }
    return symbolic_shape_.value()[i];
}


ShapeSymbol SymbolicShape::at(size_t i) const {
    if (!symbolic_shape_.has_value()) {
        AM_THROW(RuntimeError) << "Rank isn't fixed";
    }

    if (i >= symbolic_shape_->size()) {
        AM_THROW(OutOfRangeError) << "Out of range";
    }
    return symbolic_shape_.value()[i];
}

std::optional<size_t> SymbolicShape::rank() const {
    if (symbolic_shape_.has_value()) {
        return symbolic_shape_->size();
    }
    return std::nullopt;
}


std::optional<std::vector<bool>> SymbolicShape::GetSymbolicDims() const {
    const auto rank_opt = rank();
    if (!rank_opt.has_value()) {
        return std::nullopt;
    }

    const auto n = rank_opt.value();
    std::vector<bool> is_symbolic_dims;
    is_symbolic_dims.reserve(n);
    for (const auto& s: symbolic_shape_.value()) {
        is_symbolic_dims.push_back(!s.IsStatic());
    }
    return is_symbolic_dims;
}

bool SymbolicShape::IsComplete() const {
    if (!symbolic_shape_.has_value()) {
        return false;
    }

    auto shape = symbolic_shape_.value();
    return std::ranges::all_of(shape,
                               [](const ShapeSymbol& s) { return s.IsStatic(); });
}

SymbolicShape SymbolicShape::Merge(const SymbolicShape& other) const {
    if (!symbolic_shape_.has_value() ||
        !other.symbolic_shape_.has_value() ||
        symbolic_shape_->size() != other.symbolic_shape_->size()) {
        return {};
    }

    const auto n = symbolic_shape_->size();
    std::vector<ShapeSymbol> dims(n);
    for (size_t i = 0; i < n; ++i) {
        dims[i] = MergePrimitiveValue(symbolic_shape_.value()[i], other.symbolic_shape_.value()[i]);
    }
    return SymbolicShape(std::move(dims));
}

void SymbolicShape::Dump() const {
    std::cout << *this << std::endl;
}
}// namespace aethermind
