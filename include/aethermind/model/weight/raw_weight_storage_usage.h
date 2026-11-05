#ifndef AETHERMIND_MODEL_WEIGHT_RAW_WEIGHT_STORAGE_USAGE_H
#define AETHERMIND_MODEL_WEIGHT_RAW_WEIGHT_STORAGE_USAGE_H

/// @file raw_weight_storage_usage.h
/// @brief Ownership-aware raw model backing and referenced view accounting.

#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>

namespace aethermind {

struct ResolvedModelWeights;

/// @brief Visible raw backing extents and the union of referenced view ranges.
///
/// Backing extents are counted once per shared ownership identity. Referenced
/// view bytes merge overlapping and tied ranges only within the same owner.
/// Metadata, allocator rounding and resident memory are not measured here.
struct RawWeightStorageUsage {
    size_t referenced_view_bytes = 0;
    size_t mapped_bytes = 0;
    size_t owned_bytes = 0;
    size_t external_bytes = 0;
    /// False when any nonempty storage owner cannot describe its footprint.
    /// Known categories above remain partial observations, not complete totals.
    bool backing_usage_complete = true;
};

/// @brief Computes storage usage for all populated roles retained by a resolved weight set.
/// @param weights Resolved roles, including optional lm_head and every layer.
/// @return Checked accounting, InvalidArgument for malformed views or conflicting
///         owner metadata, Overflow for unrepresentable ranges/totals, or
///         ResourceExhausted for metadata allocation failure.
/// @note Exact default-empty slots are ignored so partial low-level graphs are
///       supported. Unknown footprint metadata sets backing_usage_complete=false.
///       No tensor payload is read; this cold-path query may allocate metadata.
StatusOr<RawWeightStorageUsage> ComputeRawWeightStorageUsage(
        const ResolvedModelWeights& weights) noexcept;

/// @brief Computes storage usage for roles matching an exact element dtype.
/// @param weights Resolved roles; all populated views are metadata-validated.
/// @param dtype_filter Defined element dtype with a fixed, positive lane count.
/// @return Checked accounting for matching views, or the same errors as the
///         unfiltered overload. An invalid dtype filter returns InvalidArgument.
/// @note Selected owners contribute their full backing extent once, even when
///       unselected views share that backing. Unselected owners do not affect
///       totals or completeness. No matching views yields complete zero usage.
///       No tensor payload is read; this cold-path query may allocate metadata.
StatusOr<RawWeightStorageUsage> ComputeRawWeightStorageUsage(
        const ResolvedModelWeights& weights, DataType dtype_filter) noexcept;

} // namespace aethermind

#endif
