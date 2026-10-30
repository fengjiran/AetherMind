#include "aethermind/model/weight/model_weight_normalization.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/model/loaded_model.h"
#include "utils/overflow_check.h"

#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

class NormalizedRawStorage final : public RawStorage {
public:
    explicit NormalizedRawStorage(Buffer buffer) noexcept : buffer_(std::move(buffer)) {}

    std::optional<RawStorageFootprint> footprint() const noexcept override {
        return RawStorageFootprint{RawStorageKind::kOwned, buffer_.nbytes()};
    }

private:
    Buffer buffer_;
};

struct SourceKey {
    std::shared_ptr<const RawStorage> owner;
    const std::byte* data;
    size_t bytes;
    bool bfloat;
    std::vector<int64_t> shape;
};

struct SourceLess {
    bool operator()(const SourceKey& lhs, const SourceKey& rhs) const noexcept {
        const std::owner_less<std::shared_ptr<const RawStorage>> owner_less;
        if (owner_less(lhs.owner, rhs.owner)) return true;
        if (owner_less(rhs.owner, lhs.owner)) return false;
        const auto lhs_address = reinterpret_cast<uintptr_t>(lhs.data);
        const auto rhs_address = reinterpret_cast<uintptr_t>(rhs.data);
        if (lhs_address != rhs_address) return lhs_address < rhs_address;
        if (lhs.bytes != rhs.bytes) return lhs.bytes < rhs.bytes;
        if (lhs.bfloat != rhs.bfloat) return lhs.bfloat < rhs.bfloat;
        return lhs.shape < rhs.shape;
    }
};

struct Conversion {
    const std::byte* source;
    size_t source_bytes;
    size_t target_bytes;
    bool bfloat;
    std::vector<RawWeightView*> targets;
};

struct WeightSlot {
    RawWeightView* view;
    std::string location;
};

std::vector<WeightSlot> CollectViews(ResolvedModelWeights& weights) {
    std::vector<WeightSlot> views{{&weights.embed_tokens, "embed_tokens"},
                                  {&weights.final_norm, "final_norm"}};
    if (weights.lm_head.has_value()) views.push_back({&*weights.lm_head, "lm_head"});
    for (size_t index = 0; index < weights.layers.size(); ++index) {
        auto& layer = weights.layers[index];
        for (const auto& [view, name]: {
                     std::pair{&layer.norm.input_rmsnorm, "input_rmsnorm"},
                     std::pair{&layer.norm.post_attn_rmsnorm, "post_attn_rmsnorm"},
                     std::pair{&layer.attn.q_proj, "q_proj"},
                     std::pair{&layer.attn.k_proj, "k_proj"},
                     std::pair{&layer.attn.v_proj, "v_proj"},
                     std::pair{&layer.attn.o_proj, "o_proj"},
                     std::pair{&layer.mlp.gate_proj, "gate_proj"},
                     std::pair{&layer.mlp.up_proj, "up_proj"},
                     std::pair{&layer.mlp.down_proj, "down_proj"}}) {
            views.push_back({view, "layer " + std::to_string(index) + " " + name});
        }
    }
    return views;
}

Status ValidateSource(const RawWeightView& view) {
    if (!view.IsValid()) {
        return Status::InvalidArgument("FP32 weight normalization requires a valid source weight");
    }
    // Matching-code dtype equality calls lanes(); guard its scalar precondition
    // before equality or ValidateRawWeightView can inspect an encoded vector.
    if (view.dtype.raw_lanes() != 1 ||
        (view.dtype != DataType::Float32() && view.dtype != DataType::Float(16) &&
         view.dtype != DataType::BFloat(16))) {
        return Status::Unimplemented("FP32 weight normalization requires scalar Float32, Float16, or BFloat16 weights");
    }
    AM_RETURN_IF_ERROR(ValidateRawWeightView(view));
    if (!view.is_contiguous) {
        return Status::InvalidArgument("FP32 weight normalization requires contiguous source weights");
    }
    return Status::Ok();
}

uint32_t WidenBits(uint16_t bits, bool bfloat) noexcept {
    if (bfloat) {
        uint32_t result = static_cast<uint32_t>(bits) << 16U;
        if ((bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0) {
            result |= 0x00400000U;
        }
        return result;
    }

    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000U) << 16U;
    const uint32_t exponent = (bits >> 10U) & 0x1fU;
    uint32_t mantissa = bits & 0x03ffU;
    if (exponent == 0) {
        if (mantissa == 0) return sign;
        const unsigned shift = std::countl_zero(mantissa) - 21U;
        mantissa <<= shift;
        return sign | ((113U - shift) << 23U) | ((mantissa & 0x03ffU) << 13U);
    }
    if (exponent == 0x1fU) {
        return sign | 0x7f800000U | (mantissa << 13U) |
               (mantissa == 0 ? 0U : 0x00400000U);
    }
    return sign | ((exponent + 112U) << 23U) | (mantissa << 13U);
}

Status AddBytes(size_t amount, size_t& total) {
    if (CheckOverflowAdd(total, amount, &total)) {
        return Status::Overflow("FP32 weight normalization payload size overflowed size_t");
    }
    return Status::Ok();
}

} // namespace

StatusOr<std::unique_ptr<LoadedModel>> NormalizeLoadedModelWeightsToFp32(
        const LoadedModel& model,
        Allocator& allocator,
        const ModelWeightNormalizationOptions& options) try {
    if (!allocator.device().is_cpu()) {
        return Status::InvalidArgument("FP32 weight normalization requires a CPU allocator");
    }

    ResolvedModelWeights normalized = model.GetResolvedWeights();
    const auto views = CollectViews(normalized);
    std::map<SourceKey, size_t, SourceLess> conversion_indices;
    std::vector<Conversion> conversions;
    ModelWeightNormalizationUsage usage;
    for (const auto& slot: views) {
        auto* view = slot.view;
        if (const auto status = ValidateSource(*view); !status.ok()) {
            return status.WithMessage("FP32 normalization weight " + slot.location + ": " + status.message());
        }
        if (view->dtype == DataType::Float32()) continue;

        size_t target_bytes = 0;
        if (CheckOverflowMul(view->bytes / sizeof(uint16_t), sizeof(float), &target_bytes)) {
            return Status::Overflow("FP32 normalized weight byte size overflowed size_t");
        }
        SourceKey key{view->storage, view->data, view->bytes,
                      view->dtype == DataType::BFloat(16), view->shape};
        auto [entry, inserted] = conversion_indices.try_emplace(std::move(key), conversions.size());
        if (inserted) {
            AM_RETURN_IF_ERROR(AddBytes(view->bytes, usage.converted_source_bytes));
            AM_RETURN_IF_ERROR(AddBytes(target_bytes, usage.converted_weight_bytes));
            conversions.push_back({view->data, view->bytes, target_bytes, view->dtype == DataType::BFloat(16), {}});
        }
        conversions[entry->second].targets.push_back(view);
    }

    AM_ASSIGN_OR_RETURN(usage.source_weights, SummarizeModelWeightStorage(model.GetResolvedWeights()));
    usage.peak_owned_weight_bytes = usage.source_weights.owned_bytes;
    AM_RETURN_IF_ERROR(AddBytes(usage.converted_weight_bytes, usage.peak_owned_weight_bytes));
    usage.peak_owned_weight_bytes_complete = usage.source_weights.backing_complete;

    // Converted roles no longer retain their source owners. Summarize only F32
    // roles to count backing that survives alongside the new converted payload.
    ResolvedModelWeights retained = normalized;
    for (const auto& [view, location]: CollectViews(retained)) {
        if (view->dtype != DataType::Float32()) *view = {};
    }
    AM_ASSIGN_OR_RETURN(auto retained_summary, SummarizeModelWeightStorage(retained));
    size_t final_owned_bytes = retained_summary.owned_bytes;
    AM_RETURN_IF_ERROR(AddBytes(usage.converted_weight_bytes, final_owned_bytes));
    if (options.max_owned_weight_bytes != 0) {
        if (!retained_summary.backing_complete) {
            return Status::FailedPrecondition("Finite normalized owned-weight cap requires complete retained Float32 backing metadata");
        }
        if (final_owned_bytes > options.max_owned_weight_bytes) {
            return Status::ResourceExhausted("Normalized owned-weight payload requires " +
                                             std::to_string(final_owned_bytes) + " bytes; limit " +
                                             std::to_string(options.max_owned_weight_bytes));
        }
    }

    for (const auto& conversion: conversions) {
        AM_ASSIGN_OR_RETURN(auto buffer, allocator.TryAllocate({conversion.target_bytes, 64}));
        auto* destination = static_cast<std::byte*>(buffer.mutable_data());
        for (size_t source_offset = 0, target_offset = 0;
             source_offset < conversion.source_bytes;
             source_offset += sizeof(uint16_t), target_offset += sizeof(float)) {
            uint16_t bits;
            std::memcpy(&bits, conversion.source + source_offset, sizeof(bits));
            const uint32_t widened = WidenBits(bits, conversion.bfloat);
            // memcpy implicitly creates destination float objects without
            // relying on a reduced-precision wrapper's alignment or lifetime.
            const float value = std::bit_cast<float>(widened);
            std::memcpy(destination + target_offset, &value, sizeof(value));
        }
        auto owner = std::make_shared<NormalizedRawStorage>(std::move(buffer));
        for (auto* target: conversion.targets) {
            target->data = destination;
            target->bytes = conversion.target_bytes;
            target->dtype = DataType::Float32();
            target->storage = owner;
            AM_RETURN_IF_ERROR(ValidateRawWeightView(*target));
        }
    }
    AM_ASSIGN_OR_RETURN(usage.normalized_weights, SummarizeModelWeightStorage(normalized));
    return std::make_unique<LoadedModel>(model.GetConfig(), std::move(normalized), usage);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
