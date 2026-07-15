#include "aethermind/model/graph/const_evaluator.h"
#include "aethermind/utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace aethermind {

// ── Shape/stride helpers (shared by const evaluator and folding pass) ──
// Extracts a static shape vector from a TensorSpec, rejecting dynamic/rankless shapes.
StatusOr<std::vector<int64_t>> ExtractStaticShape(const TensorSpec& spec) {
    if (!spec.shape.IsStatic()) {
        return Status::Unimplemented("requires static tensor shape");
    }

    const auto rank = spec.shape.rank();
    if (!rank.has_value()) {
        return Status::Unimplemented("requires ranked tensor shape");
    }

    std::vector<int64_t> shape(*rank);
    for (size_t i = 0; i < *rank; ++i) {
        shape[i] = spec.shape[i].GetStaticValue();
    }
    return shape;
}

// Product of shape dimensions with overflow protection against absurdly
// large tensors (e.g. shapes with huge strides that would overflow int64_t).
StatusOr<int64_t> CountElements(std::span<const int64_t> shape) {
    int64_t numel = 1;
    for (const int64_t dim: shape) {
        if (dim < 0) {
            return Status::InvalidArgument(
                    "static tensor dimensions must be non-negative");
        }

        int64_t product = 0;
        if (CheckOverflowMul(numel, dim, &product)) {
            return Status::Overflow("tensor element count overflow");
        }
        numel = product;
    }
    return numel;
}

// Chains ExtractStaticShape → CountElements → element-byte multiply,
// with overflow checks at each step.
StatusOr<size_t> CountBytes(const TensorSpec& spec) {
    auto shape = ExtractStaticShape(spec);
    AM_RETURN_IF_ERROR(shape.status());

    auto numel = CountElements(*shape);
    AM_RETURN_IF_ERROR(numel.status());

    const auto element_bytes = static_cast<size_t>(spec.dtype.nbytes());
    const auto element_count = static_cast<size_t>(*numel);
    size_t total_bytes = 0;
    if (CheckOverflowMul(element_count, element_bytes, &total_bytes)) {
        return Status::Overflow("tensor byte size overflow");
    }
    return total_bytes;
}

// Row-major (C-style) contiguous strides with overflow protection.
// For an N-D shape [d0, d1, ..., d_{N-1}], produces strides
// [d1*...*d_{N-1}, d2*...*d_{N-1}, ..., 1].
StatusOr<std::vector<int64_t>> MakeContiguousStrides(std::span<const int64_t> shape) {
    if (shape.empty()) {
        return std::vector<int64_t>{};
    }

    std::vector<int64_t> strides(shape.size());
    strides.back() = 1;
    for (size_t i = shape.size() - 1U; i > 0U; --i) {
        int64_t product = 0;
        if (CheckOverflowMul(strides[i], shape[i], &product)) {
            return Status::Overflow("contiguous stride computation overflow");
        }
        strides[i - 1U] = product;
    }
    return strides;
}

}// namespace aethermind
