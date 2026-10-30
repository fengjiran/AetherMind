#ifndef AETHERMIND_MODEL_WEIGHT_MODEL_WEIGHT_STORAGE_SUMMARY_H
#define AETHERMIND_MODEL_WEIGHT_MODEL_WEIGHT_STORAGE_SUMMARY_H

/// @file model_weight_storage_summary.h
/// @brief Ownership-aware raw model backing and logical payload accounting.

#include "aethermind/base/status.h"

#include <cstddef>

namespace aethermind {

struct ResolvedModelWeights;

/// @brief Visible raw backing extents and the union of referenced view ranges.
///
/// Backing extents are counted once per shared ownership identity. Logical
/// view bytes merge overlapping and tied ranges only within the same owner.
/// Metadata, allocator rounding and resident memory are not measured here.
struct ModelWeightStorageSummary {
    size_t logical_view_bytes = 0;
    size_t mapped_bytes = 0;
    size_t owned_bytes = 0;
    size_t external_bytes = 0;
    /// False when any nonempty storage owner cannot describe its footprint.
    /// Known categories above remain partial observations, not complete totals.
    bool backing_complete = true;
};

/// @brief Summarizes all populated roles retained by a resolved weight set.
/// @param weights Resolved roles, including optional lm_head and every layer.
/// @return Checked accounting, InvalidArgument for malformed views or conflicting
///         owner metadata, Overflow for unrepresentable ranges/totals, or
///         ResourceExhausted for metadata allocation failure.
/// @note Exact default-empty slots are ignored so partial low-level graphs are
///       supported. Unknown footprint metadata sets backing_complete=false.
///       No tensor payload is read; this cold-path query may allocate metadata.
StatusOr<ModelWeightStorageSummary> SummarizeModelWeightStorage(
        const ResolvedModelWeights& weights) noexcept;

} // namespace aethermind

#endif
