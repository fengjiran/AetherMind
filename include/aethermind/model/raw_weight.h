#ifndef AETHERMIND_MODEL_RAW_WEIGHT_H
#define AETHERMIND_MODEL_RAW_WEIGHT_H

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"
#include "utils/overflow_check.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

/// @brief Storage category reported by a raw weight backing owner.
enum class RawStorageKind : uint8_t {
    kMapped,
    kOwned,
    kExternal,
};

/// @brief Full visible backing extent, excluding allocator overhead and RSS.
///
/// Every view sharing one ownership identity must report the same extent.
/// Mapped bytes include the complete retained mapping, not only tensor payloads.
struct RawStorageFootprint {
    RawStorageKind kind = RawStorageKind::kExternal;
    size_t bytes = 0;

    AM_NODISCARD friend bool operator==(const RawStorageFootprint&,
                                        const RawStorageFootprint&) = default;
};

struct RawStorage {
    virtual ~RawStorage() = default;

    /// @brief Reports immutable backing metadata, or unknown for custom storage.
    /// @note No payload access or ownership escapes. The result must remain
    ///       unchanged for this owner's lifetime, including shared-owner aliases.
    AM_NODISCARD virtual std::optional<RawStorageFootprint> footprint() const noexcept {
        return std::nullopt;
    }
};

struct RawWeightView {
    const std::byte* data = nullptr;
    size_t bytes = 0;
    DataType dtype{};
    std::vector<int64_t> shape{};
    std::shared_ptr<const RawStorage> storage{};
    bool is_contiguous = true;

    /// @brief Checks storage presence and a fixed, positive-size element dtype.
    /// @note Shape/byte agreement is checked by ValidateRawWeightView. Empty
    ///       tensors remain valid; zero-byte element dtypes do not.
    AM_NODISCARD bool IsValid() const noexcept {
        // Inspect the encoding before any dtype equality or size query, since
        // those operations require a fixed lane count. This also rejects the
        // reserved -1 encoding, which IsScalableVector() does not classify.
        return storage != nullptr && dtype.bits() > 0 &&
               dtype.raw_lanes() > 0 &&
               dtype.raw_lanes() <= std::numeric_limits<int16_t>::max() &&
               (data != nullptr || bytes == 0);
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
/// Element dtypes must have fixed positive lanes and a nonzero byte size;
/// a zero element count is still supported.
///
/// @param view Raw weight view to validate.
/// @return Ok, or InvalidArgument describing the first violation.
inline Status ValidateRawWeightView(const RawWeightView& view) noexcept {
    if (!view.IsValid()) {
        return Status::InvalidArgument(
                "RawWeightView is not a valid view");
    }

    const int element_bytes = view.dtype.nbytes();
    if (element_bytes <= 0) {
        return Status::InvalidArgument(
                "RawWeightView requires a nonzero element byte size");
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
    if (CheckOverflowMul(numel, static_cast<uint64_t>(element_bytes),
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
