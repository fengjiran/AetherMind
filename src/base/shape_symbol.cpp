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
        return os << "[?]";
    }

    const auto& shape = *s.shape();
    const auto rank = *rank_opt;
    os << "[";
    for (size_t i = 0; i < rank; ++i) {
        os << shape[i];
        if (i != rank - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        symbolic_shape_ = std::vector<ShapeSymbol>(*rank);
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& shape) {
    std::vector<ShapeSymbol> symbolic_shape;
    symbolic_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        if (shape[i].has_value()) {
            symbolic_shape.emplace_back(ShapeSymbol::CreateFromValue(*shape[i]));
        } else {
            symbolic_shape.emplace_back(ShapeSymbol::Unknown());
        }
    }
    symbolic_shape_ = std::move(symbolic_shape);
}

SymbolicShape::SymbolicShape(IntArrayView shape) {
    std::vector<ShapeSymbol> symbolic_shape;
    symbolic_shape.reserve(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        symbolic_shape.emplace_back(ShapeSymbol::CreateFromValue(shape[i]));
    }
    symbolic_shape_ = std::move(symbolic_shape);
}

const ShapeSymbol& SymbolicShape::operator[](size_t i) const {
    AM_CHECK(IsRanked(), "Cannot access dimensions of an Unranked shape.");
    AM_CHECK(i < symbolic_shape_->size(), "Dimension index out of bounds.");
    return (*symbolic_shape_)[i];
}

ShapeSymbol& SymbolicShape::operator[](size_t i) {
    AM_CHECK(IsRanked(), "Cannot access dimensions of an Unranked shape.");
    AM_CHECK(i < symbolic_shape_->size(), "Dimension index out of bounds.");
    return (*symbolic_shape_)[i];
}

std::optional<size_t> SymbolicShape::rank() const noexcept {
    if (IsRanked()) {
        return symbolic_shape_->size();
    }
    return std::nullopt;
}

std::optional<std::vector<bool>> SymbolicShape::GetSymbolicDims() const {
    const auto rank_opt = rank();
    if (!rank_opt.has_value()) {
        return std::nullopt;
    }

    std::vector<bool> is_symbolic_dims;
    is_symbolic_dims.reserve(*rank_opt);
    for (const auto& s: *symbolic_shape_) {
        is_symbolic_dims.push_back(!s.IsStatic());
    }
    return is_symbolic_dims;
}

bool SymbolicShape::IsStatic() const noexcept {
    if (!IsRanked()) {
        return false;
    }
    return std::ranges::all_of(*symbolic_shape_, [](const ShapeSymbol& s) { return s.IsStatic(); });
}

SymbolicShape SymbolicShape::Merge(const SymbolicShape& other) const {
    if (!IsRanked() || !other.IsRanked() ||
        symbolic_shape_->size() != other.symbolic_shape_->size()) {
        return {};
    }

    const auto n = symbolic_shape_->size();
    std::vector<ShapeSymbol> dims;
    dims.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        dims.emplace_back(MergePrimitiveValue((*symbolic_shape_)[i], (*other.symbolic_shape_)[i]));
    }
    return SymbolicShape(std::move(dims));
}

void SymbolicShape::Dump() const {
    std::cout << *this << std::endl;
}
}// namespace aethermind
