#ifndef AETHERMIND_MODEL_RAW_WEIGHT_H
#define AETHERMIND_MODEL_RAW_WEIGHT_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

struct RawStorage {
    virtual ~RawStorage() = default;
};

struct RawWeightView {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawStorage> storage{};
    bool is_contiguous = true;

    AM_NODISCARD bool IsValid() const noexcept {
        return storage != nullptr && dtype.bits() > 0 && (data != nullptr || bytes == 0);
    }

    AM_NODISCARD bool IsAligned(size_t alignment) const noexcept {
        return alignment != 0 && data != nullptr &&
               reinterpret_cast<std::uintptr_t>(data) % alignment == 0;
    }
};

/// @brief Validates that a raw weight view's byte size exactly matches the
/// byte size implied by its logical shape and dtype.
///
/// Weight packing derives its copy size from shape × dtype byte size, so any
/// mismatch would read past the backing storage (undersized bytes) or
/// silently corrupt fused composite layouts (oversized bytes). Every
/// dimension must be non-negative and the shape product checked for overflow.
///
/// @param view Raw weight view to validate.
/// @return Ok, or InvalidArgument describing the first violation.
AM_NODISCARD inline Status ValidateRawWeightView(const RawWeightView& view) noexcept {
    if (!view.IsValid()) {
        return Status::InvalidArgument(
                "RawWeightView is not a valid view");
    }

    uint64_t numel = 1;
    for (const int64_t dim: view.shape) {
        if (dim < 0) {
            return Status::InvalidArgument(
                    "RawWeightView shape has a negative dimension");
        }
        if (CheckOverflowMul(numel, static_cast<uint64_t>(dim), &numel)) {
            return Status::InvalidArgument(
                    "RawWeightView shape product overflows");
        }
    }

    uint64_t expected_bytes = 0;
    if (CheckOverflowMul(numel,
                         static_cast<uint64_t>(view.dtype.nbytes()),
                         &expected_bytes)) {
        return Status::InvalidArgument(
                "RawWeightView byte size overflows");
    }

    if (view.bytes != expected_bytes) {
        return Status::InvalidArgument(
                "RawWeightView byte size does not match its logical shape");
    }
    return Status::Ok();
}

using RawWeightTable = std::unordered_map<std::string, RawWeightView>;

} // namespace aethermind

#endif
