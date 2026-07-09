#include "aethermind/shape_inference/shape_symbol.h"
#include "error.h"

#include <algorithm>

namespace aethermind {

std::atomic<int64_t> ShapeSymbol::next_symbol_ = -2;

SymbolicShape::SymbolicShape(std::optional<size_t> rank) {
    if (rank.has_value()) {
        symbolic_shape_ = std::vector<ShapeSymbol>(*rank);
    }
}

SymbolicShape::SymbolicShape(const std::vector<std::optional<int64_t>>& shape) {
    std::vector<ShapeSymbol> symbolic_shape;
    symbolic_shape.reserve(shape.size());
    for (auto i: shape) {
        if (i.has_value()) {
            symbolic_shape.emplace_back(ShapeSymbol::CreateFromValue(*i));
        } else {
            symbolic_shape.emplace_back(ShapeSymbol::Unknown());
        }
    }
    symbolic_shape_ = std::move(symbolic_shape);
}

SymbolicShape::SymbolicShape(IntArrayView shape) {
    std::vector<ShapeSymbol> symbolic_shape;
    symbolic_shape.reserve(shape.size());
    for (const auto i: shape) {
        symbolic_shape.emplace_back(ShapeSymbol::CreateFromValue(i));
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
    return std::ranges::all_of(*symbolic_shape_,
                               [](const ShapeSymbol& s) { return s.IsStatic(); });
}

SymbolicShape SymbolicShape::Join(const SymbolicShape& other) const {
    if (!IsRanked() || !other.IsRanked() || rank() != other.rank()) {
        return {};
    }

    const auto n = rank();
    std::vector<ShapeSymbol> dims;
    dims.reserve(*n);
    for (size_t i = 0; i < n; ++i) {
        dims.emplace_back(JoinShapeSymbol((*this)[i], other[i]));
    }
    return SymbolicShape(std::move(dims));
}

StatusOr<SymbolicShape> SymbolicShape::Unify(const SymbolicShape& other) const {
    if (IsUnranked()) return other;
    if (other.IsUnranked()) return *this;

    if (rank() != other.rank()) {
        return Status::InvalidArgument("Unification Failed: Rank mismatch.");
    }

    const auto n = rank();
    std::vector<ShapeSymbol> dims;
    dims.reserve(*n);
    for (size_t i = 0; i < n; ++i) {
        auto unified_dim_or = UnifyShapeSymbol((*this)[i], other[i]);
        if (!unified_dim_or.ok()) {
            return unified_dim_or.status();
        }

        dims.emplace_back(unified_dim_or.value());
    }
    return SymbolicShape(std::move(dims));
}

void SymbolicShape::Dump() const {
    std::cout << *this << std::endl;
}

ShapeSymbol JoinShapeSymbol(const ShapeSymbol& a, const ShapeSymbol& b) {
    if (a == b) {
        // Handles all three identity cases uniformly:
        // 1. Same static dimension (4096 == 4096)
        // 2. Same dynamic constraint (S2 == S2) — preserves symbolic identity
        // 3. Same fully-unknown placeholder (? == ?)
        return a;
    }
    // Any divergence (e.g. 16 vs 32, S2 vs S3, 16 vs ?) means the dimension
    // is shape-inconsistent across branches. Relax the constraint by
    // assigning a fresh tracking symbol.
    return ShapeSymbol::Create();
}

StatusOr<ShapeSymbol> UnifyShapeSymbol(const ShapeSymbol& a, const ShapeSymbol& b) {
    // 1. Fast path: identical values (Static==Static, Sym==Sym, Unknown==Unknown)
    if (a == b) {
        return a;
    }

    // 2. Hard conflict: both are static but with different values.
    if (a.IsStatic() && b.IsStatic()) {
        return Status::InvalidArgument(
                "Shape Unification Failed: incompatible static dimensions.");
    }

    // 3. Static overrides: if exactly one is static, it is the concrete value
    //    and the other is either symbolic or unknown.
    if (a.IsStatic()) {
        return a;
    }

    if (b.IsStatic()) {
        return b;
    }

    // 4. Unknown has the lowest priority — resolve to the other symbol.
    if (a.IsUnknown()) {
        return b;
    }

    if (b.IsUnknown()) {
        return a;
    }

    // 5. Symbolic collision: both are symbolic but distinct (e.g. S2 vs S3).
    // TODO(Future): Introduce SymbolConstraintSolver to record a == b
    // as an equality constraint.
    //
    // Without a solver, the safest behaviour is to degrade to Unknown.
    // This signals that the engine can no longer track the equality
    // precisely at compile time and will rely on runtime checks.
    // (Returning a or b would preserve the local link but is misleading
    // under strict semantics.)
    return ShapeSymbol::Unknown();
}

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
}// namespace aethermind
