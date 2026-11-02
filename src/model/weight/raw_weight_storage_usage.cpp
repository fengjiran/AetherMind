#include "aethermind/model/weight/raw_weight_storage_usage.h"
#include "aethermind/model/resolved_model_weights.h"
#include "utils/overflow_check.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <new>
#include <optional>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

struct AddrRange {
    uintptr_t begin;
    uintptr_t end;
};

struct OwnerStorageMetadata {
    std::optional<RawStorageFootprint> footprint;
    std::vector<AddrRange> referenced_ranges;
};

using OwnerMap = std::map<std::shared_ptr<const RawStorage>, OwnerStorageMetadata,
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
        return Status::InvalidArgument(
                "Raw weight storage usage requires a fixed element dtype");
    }

    AM_RETURN_IF_ERROR(ValidateRawWeightView(view));
    if (!view.is_contiguous) {
        return Status::InvalidArgument(
                "Raw weight storage usage requires contiguous views");
    }

    const auto footprint = view.storage->footprint();
    auto [owner, inserted] = owners.try_emplace(
            view.storage, OwnerStorageMetadata{footprint, {}});
    if (!inserted && owner->second.footprint != footprint) {
        return Status::InvalidArgument(
                "Shared raw weight owner reports conflicting footprint metadata");
    }

    if (view.bytes == 0) {
        return Status::Ok();
    }

    const auto begin = reinterpret_cast<uintptr_t>(view.data);
    uintptr_t end = 0;
    if (!std::in_range<uintptr_t>(view.bytes) ||
        CheckOverflowAdd(begin, view.bytes, &end)) {
        return Status::Overflow(
                "Raw weight address range overflowed uintptr_t");
    }
    owner->second.referenced_ranges.push_back({begin, end});
    return Status::Ok();
}

StatusOr<size_t> UnionBytes(std::vector<AddrRange>& ranges) {
    std::ranges::sort(ranges,
                      [](const auto& lhs, const auto& rhs) {
                          return lhs.begin < rhs.begin ||
                                 (lhs.begin == rhs.begin && lhs.end < rhs.end);
                      });

    size_t bytes = 0;
    const auto add_range = [&bytes](const AddrRange& range) -> Status {
        size_t extent = 0;
        if (CheckNarrow(range.end - range.begin, &extent) ||
            CheckOverflowAdd(bytes, extent, &bytes)) {
            return Status::Overflow(
                    "Raw weight referenced range union overflowed size_t");
        }
        return Status::Ok();
    };

    std::optional<AddrRange> current;
    for (const auto& range: ranges) {
        if (!current.has_value()) {
            current = range;
        } else if (range.begin <= current->end) {
            current->end = std::max(current->end, range.end);
        } else {
            AM_RETURN_IF_ERROR(add_range(*current));
            current = range;
        }
    }

    if (current.has_value()) {
        AM_RETURN_IF_ERROR(add_range(*current));
    }
    return bytes;
}

} // namespace

StatusOr<RawWeightStorageUsage> ComputeRawWeightStorageUsage(
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

    RawWeightStorageUsage usage;
    for (auto& [owner, retained]: owners) {
        AM_ASSIGN_OR_RETURN(const size_t referenced_bytes,
                            UnionBytes(retained.referenced_ranges));
        if (CheckOverflowAdd(usage.referenced_view_bytes, referenced_bytes, &usage.referenced_view_bytes)) {
            return Status::Overflow(
                    "Raw model referenced view bytes overflowed size_t");
        }

        if (!retained.footprint.has_value()) {
            usage.backing_usage_complete = false;
            continue;
        }

        const auto [kind, bytes] = *retained.footprint;
        if (referenced_bytes > bytes) {
            return Status::InvalidArgument(
                    "Raw owner footprint is smaller than its referenced view ranges");
        }

        size_t* category = nullptr;
        switch (kind) {
            case RawStorageKind::kMapped:
                category = &usage.mapped_bytes;
                break;
            case RawStorageKind::kOwned:
                category = &usage.owned_bytes;
                break;
            case RawStorageKind::kExternal:
                category = &usage.external_bytes;
                break;
            default:
                return Status::InvalidArgument(
                        "Unknown raw storage footprint kind");
        }

        if (CheckOverflowAdd(*category, bytes, category)) {
            return Status::Overflow(
                    "Raw model backing bytes overflowed size_t");
        }
    }
    return usage;
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
