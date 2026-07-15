// Shape and stride validation, assignment, and query implementation.
//
// ShapeAndStride is the canonical metadata descriptor owned by every Tensor.
// This file implements the mutation and query methods: setting metadata from
// caller-provided shape/strides, computing contiguous strides, counting
// elements, checking contiguity, and computing the maximum flat offset for
// buffer sizing.
//
// A default-constructed ShapeAndStride is uninitialized (is_initialized() is
// false, numel() returns 0, is_contiguous() returns false).  After a
// successful set() or set_contiguous() call the metadata is initialized;
// an explicit empty shape (size 0) is a valid rank-0 state with numel = 1
// and is_contiguous = true.
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/utils/overflow_check.h"

#include <limits>

namespace aethermind {

// Copies validated shape and strides into internal storage.
//
// All validation completes before mutating state: rank match, rank bound,
// and non-negative dimensions.  Once validation passes, the metadata is
// marked initialized, the caller-provided values are copied, and any
// trailing slots beyond the new rank are zeroed for deterministic access.
void ShapeAndStride::set(IntArrayView shape, IntArrayView strides) {
    AM_CHECK(shape.size() == strides.size(), "shape/strides size mismatch");
    AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");

    const auto new_size = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < new_size; ++i) {
        AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
    }

    size_ = new_size;
    initialized_ = true;
    std::ranges::copy(shape, shape_.begin());
    std::ranges::copy(strides, strides_.begin());

    for (uint32_t i = size_; i < kMaxRank; ++i) {
        shape_[i] = 0;
        strides_[i] = 0;
    }
}

// Accepts only shape and computes row-major contiguous strides.
//
// After validation (rank bound, non-negative dimensions), the metadata is
// marked initialized.  Rank-0 (empty shape) is set as contiguous and returns
// immediately. Otherwise, strides are computed backward from the innermost
// dimension (stride 1) with overflow checking for each multiplication.
void ShapeAndStride::set_contiguous(IntArrayView shape) {
    AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");
    auto new_size = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < new_size; ++i) {
        AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
    }

    size_ = new_size;
    initialized_ = true;
    if (size_ == 0) {
        // Rank-0: all slots zeroed, contiguous, single element.
        for (uint32_t i = 0; i < kMaxRank; ++i) {
            shape_[i] = 0;
            strides_[i] = 0;
        }
        return;
    }

    std::ranges::copy(shape, shape_.begin());

    strides_[size_ - 1] = 1;
    for (int32_t i = size_ - 2; i >= 0; --i) {
        AM_CHECK(!CheckOverflowMul(strides_[i + 1], shape_[i + 1], &strides_[i]),
                 "Stride calculation overflow");
    }

    for (uint32_t i = size_; i < kMaxRank; ++i) {
        shape_[i] = 0;
        strides_[i] = 0;
    }
}

// Returns the total number of elements described by the stored shape.
//
// Uninitialized metadata returns 0.  Rank-0 (initialized, size 0) returns 1.
// For non-empty shapes the product of dimensions is computed via
// SafeMultiplyU64 with overflow detection; an overflow or result exceeding
// the int64_t/size_t range triggers a fatal check.
int64_t ShapeAndStride::numel() const noexcept {
    if (size_ == 0) {
        return initialized_ ? 1 : 0;
    }

    uint64_t numel = 0;
    bool overflow = SafeMultiplyU64(shape(), &numel);
    constexpr auto kNumelMax = std::min<uint64_t>(
            std::numeric_limits<int64_t>::max(),
            std::numeric_limits<size_t>::max());

    overflow |= numel > kNumelMax;
    AM_CHECK(!overflow, "Integer multiplication overflow when compute numel.");
    return static_cast<int64_t>(numel);
}

// Returns true when the stored strides describe row-major contiguous layout.
//
// Uninitialized metadata returns false.  Rank-0 (initialized, size 0) is
// trivially contiguous.  For non-empty shapes, the function walks dimensions
// from innermost outward, checking that each non-1 dimension's stride matches
// the product of the dimensions inside it.  The stride product is guarded
// against overflow since a valid contiguous layout must always produce a
// stride that fits in int64_t when the corresponding shape dimension fits.
bool ShapeAndStride::is_contiguous() const noexcept {
    if (size_ == 0) {
        return initialized_;
    }

    int64_t expected_stride = 1;
    for (int i = size_ - 1; i >= 0; --i) {
        if (shape_[i] == 1) {
            continue;
        }

        if (strides_[i] != expected_stride) {
            return false;
        }

        int64_t next_expected_stride = 0;
        AM_CHECK(!CheckOverflowMul(expected_stride, shape_[i], &next_expected_stride),
                 "Stride validation overflow");
        expected_stride = next_expected_stride;
    }
    return true;
}

// Returns the maximum flat byte offset reachable by valid element indices.
//
// Rank-0 and shapes containing at least one zero dimension return 0 (no
// addressable storage beyond the base pointer).  The first pass validates
// non-negative dimensions/offsets and short-circuits on zero-element axes.
// The second pass computes sum((shape[i] - 1) * strides[i]) with per-term
// overflow and cumulative addition guards.  Callers use this result to size
// backing buffers and validate storage range.
int64_t ShapeAndStride::max_element_offset() const {
    if (size_ == 0) {
        return 0;
    }

    for (int32_t i = 0; i < size_; ++i) {
        AM_CHECK(shape_[i] >= 0);
        AM_CHECK(strides_[i] >= 0);
        if (shape_[i] == 0) {
            return 0;
        }
    }

    int64_t offset = 0;
    for (int32_t i = 0; i < size_; ++i) {
        int64_t term = 0;
        AM_CHECK(!CheckOverflowMul((shape_[i] - 1), strides_[i], &term));
        AM_CHECK(term >= 0);
        AM_CHECK(offset <= std::numeric_limits<int64_t>::max() - term, "max_element_offset overflow.");
        offset += term;
    }

    return offset;
}

}// namespace aethermind