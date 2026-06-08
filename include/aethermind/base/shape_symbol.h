#ifndef AETHERMIND_SHAPE_SYMBOL_H
#define AETHERMIND_SHAPE_SYMBOL_H

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

}// namespace aethermind

#endif// AETHERMIND_SHAPE_SYMBOL_H
