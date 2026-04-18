//
// Created by richard on 4/18/26.
//
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace aethermind {

void ShapeAndStride::set(IntArrayView shape, IntArrayView strides) {
    AM_CHECK(shape.size() == strides.size(), "shape/strides size mismatch");
    AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");

    auto new_size = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < new_size; ++i) {
        AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
    }

    size_ = new_size;
    std::ranges::copy(shape, shape_.begin());
    std::ranges::copy(strides, strides_.begin());

    for (uint32_t i = size_; i < kMaxRank; ++i) {
        shape_[i] = 0;
        strides_[i] = 0;
    }
}

void ShapeAndStride::set_contiguous(IntArrayView shape) {
    AM_CHECK(shape.size() <= static_cast<size_t>(kMaxRank), "rank exceeds kMaxRank");
    auto new_size = static_cast<int32_t>(shape.size());
    for (int32_t i = 0; i < new_size; ++i) {
        AM_CHECK(shape[i] >= 0, "shape dimensions must be non-negative");
    }

    size_ = new_size;
    if (size_ == 0) {
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

int64_t ShapeAndStride::numel() const noexcept {
    if (size_ == 0) {
        return 0;
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

bool ShapeAndStride::is_contiguous() const noexcept {
    if (size_ == 0) {
        return false;
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