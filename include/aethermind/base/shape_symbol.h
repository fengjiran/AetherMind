#ifndef AETHERMIND_SHAPE_SYMBOL_H
#define AETHERMIND_SHAPE_SYMBOL_H

#include "aethermind/base/status.h"
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
        return {};
    }

    /// Creates a ShapeSymbol from a known static dimension value.
    /// @param val A non-negative static dimension value.
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

/// @brief Joins two shape symbols, typically used for control-flow branches.
/// @return The identical symbol if they match, or a fresh new symbol if they differ.
ShapeSymbol JoinShapeSymbol(const ShapeSymbol& a, const ShapeSymbol& b);

StatusOr<ShapeSymbol> UnifyShapeSymbol(const ShapeSymbol& a, const ShapeSymbol& b);

/// Shape of a tensor represented with ShapeSymbol objects. Unranked, ranked
/// unknown dimensions, partially known, and fully known shapes are all supported.
class SymbolicShape {
public:
    SymbolicShape() noexcept = default;

    /// Constructs a SymbolicShape with known rank but all dimensions unknown.
    explicit SymbolicShape(std::optional<size_t> rank);

    /// Constructs a SymbolicShape from a mix of known and unknown dimensions.
    /// Each element is a concrete value for a static dimension, or std::nullopt
    /// for an unknown dimension.
    explicit SymbolicShape(const std::vector<std::optional<int64_t>>& shape);

    explicit SymbolicShape(std::vector<ShapeSymbol> shape) noexcept
        : symbolic_shape_(std::move(shape)) {}

    explicit SymbolicShape(IntArrayView shape);

    AM_NODISCARD bool IsRanked() const noexcept {
        return symbolic_shape_.has_value();
    }

    /// Returns rank or nullopt in case of unranked shape.
    AM_NODISCARD std::optional<size_t> rank() const noexcept;

    AM_NODISCARD const std::optional<std::vector<ShapeSymbol>>& shape() const noexcept {
        return symbolic_shape_;
    }

    AM_NODISCARD auto begin() const {
        AM_CHECK(IsRanked());
        return symbolic_shape_->begin();
    }

    AM_NODISCARD auto end() const {
        AM_CHECK(IsRanked());
        return symbolic_shape_->end();
    }

    AM_NODISCARD const ShapeSymbol& operator[](size_t i) const;
    AM_NODISCARD ShapeSymbol& operator[](size_t i);

    AM_NODISCARD std::optional<std::vector<bool>> GetSymbolicDims() const;

    AM_NODISCARD bool IsUnranked() const noexcept {
        return !symbolic_shape_.has_value();
    }

    /// Checks whether the shape is fully static, i.e. both rank and every
    /// dimension are known.
    AM_NODISCARD bool IsStatic() const noexcept;

    void Dump() const;

    /// Merges this SymbolicShape with another. Only dimensions that are both
    /// static and identical are retained. If either shape has an unknown rank,
    /// or if their ranks differ, the result is unranked.
    AM_NODISCARD SymbolicShape Merge(const SymbolicShape& other) const;

    auto operator<=>(const SymbolicShape&) const noexcept = default;

private:
    std::optional<std::vector<ShapeSymbol>> symbolic_shape_{std::nullopt};
};

std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);

}// namespace aethermind

#endif// AETHERMIND_SHAPE_SYMBOL_H
