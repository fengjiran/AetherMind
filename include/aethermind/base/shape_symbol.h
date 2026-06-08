#ifndef AETHERMIND_SHAPE_SYMBOL_H
#define AETHERMIND_SHAPE_SYMBOL_H

#include "macros.h"
#include "utils/logging.h"

#include <atomic>
#include <iosfwd>

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
    ShapeSymbol() : value_(-1) {}

    AM_NODISCARD int64_t value() const {
        return value_;
    }

    AM_NODISCARD bool IsStatic() const {
        return value_ >= 0;
    }

    AM_NODISCARD int64_t GetStaticValue() const {
        AM_CHECK(IsStatic());
        return value_;
    }

    bool operator==(const ShapeSymbol& other) const {
        return value_ == other.value_;
    }

    bool operator<(const ShapeSymbol& other) const {
        return value_ < other.value_;
    }

    static ShapeSymbol CreateFromValue(int64_t val) {
        return ShapeSymbol(val);
    }

    static ShapeSymbol Create() {
        return CreateFromValue(-static_cast<int64_t>(++num_symbols_));
    }

private:
    explicit ShapeSymbol(int64_t val) : value_(val) {}

    int64_t value_;
    // Atomic counter for generating unique symbolic IDs.
    // Thread-safe without external synchronization.
    inline static std::atomic<size_t> num_symbols_ = 1;
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
