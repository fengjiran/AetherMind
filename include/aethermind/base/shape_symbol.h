#ifndef AETHERMIND_SHAPE_SYMBOL_H
#define AETHERMIND_SHAPE_SYMBOL_H

#include "container/array_view.h"
#include "macros.h"
#include "utils/logging.h"

#include <atomic>

namespace aethermind {

/// Represents one symbolic tensor dimension.
///
/// A non-negative value is a known static dimension. A negative value is a
/// generated symbolic dimension that stands for an unknown runtime size. Two
/// generated symbols are intentionally distinct unless they carry the same
/// internal value, so equality can express both fixed-dimension equality and
/// symbolic identity. Instances are value types and thread-safe.
class ShapeSymbol {
public:
    ShapeSymbol() noexcept : value_(kUnknown) {}

    AM_NODISCARD int64_t value() const noexcept {
        return value_;
    }

    AM_NODISCARD bool IsStatic() const noexcept {
        return value_ >= 0;
    }

    AM_NODISCARD bool IsUnknown() const noexcept {
        return value_ == kUnknown;
    }

    AM_NODISCARD bool IsSymbolic() const noexcept {
        return value_ < kUnknown;
    }

    AM_NODISCARD int64_t GetStaticValue() const {
        AM_CHECK(IsStatic());
        return value_;
    }

    auto operator<=>(const ShapeSymbol&) const noexcept = default;

    /// Creates a fully unconstrained unknown dimension.
    ///
    /// Prefer `Create()` for dimensions that participate in symbolic
    /// inference. Use `Unknown()` only when the dimension carries no
    /// constraints at all, such as the output length of a NonZero operator.
    static ShapeSymbol Unknown() noexcept {
        return ShapeSymbol();
    }

    static ShapeSymbol CreateFromValue(int64_t val) {
        AM_CHECK(val >= 0 && "CreateFromValue must take a non-negative value.");
        return ShapeSymbol(val);
    }

    /// Creates a fresh symbolic dimension that participates in shape inference.
    ///
    /// Each call returns a distinct symbol, so two `Create()` results are
    /// never equal. Use this for dimensions whose value must be deduced
    /// from context during type unification.
    static ShapeSymbol Create() noexcept {
        return ShapeSymbol(next_symbol_.fetch_sub(1, std::memory_order_relaxed));
    }

private:
    explicit ShapeSymbol(int64_t val) noexcept : value_(val) {}

    static constexpr int64_t kUnknown = -1;

    static std::atomic<int64_t> next_symbol_;

    int64_t value_;
};

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);

/// Merges two shape symbols: identical static values stay static;
/// any mismatch or dynamic input produces a new dynamic symbol.
/// Used during symbolic shape unification in type inference.
inline ShapeSymbol MergePrimitiveValue(const ShapeSymbol& a, const ShapeSymbol& b) {
    if (a.IsStatic() && b.IsStatic() && a == b) {
        return a;
    }
    return ShapeSymbol::Create();
}

// Shape of a Tensor represented with ShapeSymbol's. Unranked, ranked unknown
// dims, partially known and fully known shapes are all supported.
class SymbolicShape {
public:
    SymbolicShape() = default;

    // Known rank but unknown dimensions
    explicit SymbolicShape(std::optional<size_t> rank);

    // Mix of known and unknown ranks
    explicit SymbolicShape(const std::vector<std::optional<int64_t>>& shape);

    explicit SymbolicShape(std::vector<ShapeSymbol> shape) : symbolic_shape_(std::move(shape)) {}

    explicit SymbolicShape(IntArrayView shape);

    // Returns rank or nullopt in case of unranked shape.
    AM_NODISCARD std::optional<size_t> rank() const;

    AM_NODISCARD const std::optional<std::vector<ShapeSymbol>>& shape() const noexcept {
        return symbolic_shape_;
    }

    ShapeSymbol operator[](size_t i) const;

    AM_NODISCARD ShapeSymbol at(size_t i) const;

    AM_NODISCARD std::optional<std::vector<bool>> GetSymbolicDims() const;

    AM_NODISCARD bool IsUnranked() const noexcept {
        return !symbolic_shape_.has_value();
    }

    // Checks whether the shape is fully static, i.e. rank and shape
    // of every dimension are known.
    AM_NODISCARD bool IsComplete() const;

    void Dump() const;

    // Generate a new SymbolicShape through merging itself with another SymbolicShape.
    // Only dimensions that are both static and identical will be retained.
    // If either shape has an unknown rank, or if their ranks differ,
    // the resulting shape will be unranked.
    AM_NODISCARD SymbolicShape Merge(const SymbolicShape& other) const;

    friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return lhs.symbolic_shape_ == rhs.symbolic_shape_;
    }

    friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return !(lhs == rhs);
    }

private:
    std::optional<std::vector<ShapeSymbol>> symbolic_shape_{std::nullopt};
};


std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);

}// namespace aethermind

#endif// AETHERMIND_SHAPE_SYMBOL_H
