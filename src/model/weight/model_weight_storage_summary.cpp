#include "aethermind/model/weight/model_weight_storage_summary.h"
#include "aethermind/model/resolved_model_weights.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

struct AddressRange {
    uintptr_t begin;
    uintptr_t end;
};

struct OwnerRanges {
    std::optional<RawStorageFootprint> footprint;
    std::vector<AddressRange> ranges;
};

using OwnerMap = std::map<std::shared_ptr<const RawStorage>, OwnerRanges,
                          std::owner_less<std::shared_ptr<const RawStorage>>>;

bool IsDefaultEmpty(const RawWeightView& view) noexcept {
    return view.data == nullptr && view.bytes == 0 && view.shape.empty() &&
           view.storage == nullptr && view.dtype.IsUndefined() &&
           view.dtype.bits() == 0 && view.dtype.raw_lanes() == 0 && view.is_contiguous;
}

Status AddView(OwnerMap& owners, const RawWeightView& view) {
    if (IsDefaultEmpty(view)) {
        return Status::Ok();
    }
    // ValidateRawWeightView uses nbytes(), whose fixed-lane precondition must
    // be checked before accepting externally constructed low-level metadata.
    if (view.dtype.IsUndefined() || view.dtype.bits() == 0 ||
        view.dtype.raw_lanes() == 0 ||
        view.dtype.raw_lanes() > std::numeric_limits<int16_t>::max()) {
        return Status::InvalidArgument("Raw weight storage summary requires a fixed element dtype");
    }
    AM_RETURN_IF_ERROR(ValidateRawWeightView(view));
    if (!view.is_contiguous) {
        return Status::InvalidArgument("Raw weight storage summary requires contiguous views");
    }

    const auto footprint = view.storage->footprint();
    auto [owner, inserted] = owners.try_emplace(view.storage, OwnerRanges{footprint, {}});
    if (!inserted && owner->second.footprint != footprint) {
        return Status::InvalidArgument("Shared raw weight owner reports conflicting footprint metadata");
    }

    if (view.bytes == 0) {
        return Status::Ok();
    }
    const auto begin = reinterpret_cast<uintptr_t>(view.data);
    uintptr_t end = 0;
    if (!std::in_range<uintptr_t>(view.bytes) ||
        CheckOverflowAdd(begin, static_cast<uintptr_t>(view.bytes), &end)) {
        return Status::Overflow("Raw weight address range overflowed uintptr_t");
    }
    owner->second.ranges.push_back({begin, end});
    return Status::Ok();
}

StatusOr<size_t> UnionBytes(std::vector<AddressRange>& ranges) {
    std::sort(ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.begin < rhs.begin || (lhs.begin == rhs.begin && lhs.end < rhs.end);
    });
    size_t bytes = 0;
    uintptr_t begin = 0;
    uintptr_t end = 0;
    const auto add_range = [&]() -> Status {
        size_t extent = 0;
        if (CheckNarrow(end - begin, &extent) || CheckOverflowAdd(bytes, extent, &bytes)) {
            return Status::Overflow("Raw weight logical range union overflowed size_t");
        }
        return Status::Ok();
    };
    for (const auto& range: ranges) {
        if (range.begin > end) {
            AM_RETURN_IF_ERROR(add_range());
            begin = range.begin;
            end = range.end;
        } else {
            end = std::max(end, range.end);
        }
    }
    AM_RETURN_IF_ERROR(add_range());
    return bytes;
}

} // namespace

StatusOr<ModelWeightStorageSummary> SummarizeModelWeightStorage(
        const ResolvedModelWeights& weights) noexcept try {
    OwnerMap owners;
    AM_RETURN_IF_ERROR(AddView(owners, weights.embed_tokens));
    AM_RETURN_IF_ERROR(AddView(owners, weights.final_norm));
    if (weights.lm_head.has_value()) {
        AM_RETURN_IF_ERROR(AddView(owners, *weights.lm_head));
    }
    for (const auto& layer: weights.layers) {
        for (const auto* view: {&layer.norm.input_rmsnorm, &layer.norm.post_attn_rmsnorm,
                                &layer.attn.q_proj, &layer.attn.k_proj, &layer.attn.v_proj,
                                &layer.attn.o_proj, &layer.mlp.gate_proj, &layer.mlp.up_proj,
                                &layer.mlp.down_proj}) {
            AM_RETURN_IF_ERROR(AddView(owners, *view));
        }
    }

    ModelWeightStorageSummary summary;
    for (auto& [owner, retained]: owners) {
        AM_ASSIGN_OR_RETURN(const size_t logical_bytes, UnionBytes(retained.ranges));
        if (CheckOverflowAdd(summary.logical_view_bytes, logical_bytes, &summary.logical_view_bytes)) {
            return Status::Overflow("Raw model logical view bytes overflowed size_t");
        }
        if (!retained.footprint.has_value()) {
            summary.backing_complete = false;
            continue;
        }
        const auto [kind, bytes] = *retained.footprint;
        if (logical_bytes > bytes) {
            return Status::InvalidArgument("Raw owner footprint is smaller than its referenced view ranges");
        }
        size_t* category = nullptr;
        switch (kind) {
            case RawStorageKind::kMapped:
                category = &summary.mapped_bytes;
                break;
            case RawStorageKind::kOwned:
                category = &summary.owned_bytes;
                break;
            case RawStorageKind::kExternal:
                category = &summary.external_bytes;
                break;
            default:
                return Status::InvalidArgument("Unknown raw storage footprint kind");
        }
        if (CheckOverflowAdd(*category, bytes, category)) {
            return Status::Overflow("Raw model backing bytes overflowed size_t");
        }
    }
    return summary;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
