#include "aethermind/model/weight/raw_weight_fp32_conversion.h"
#include "aethermind/dtypes/bfloat16.h"
#include "aethermind/dtypes/half.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/model/loaded_model.h"
#include "utils/overflow_check.h"

#include <cstring>
#include <map>
#include <memory>
#include <new>
#include <string>
#include <utility>
#include <vector>

namespace aethermind {
namespace {

class ConvertedRawStorage final : public RawStorage {
public:
    explicit ConvertedRawStorage(Buffer buffer) noexcept : buffer_(std::move(buffer)) {}

    AM_NODISCARD std::optional<RawStorageFootprint> footprint() const noexcept override {
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
        constexpr std::owner_less<std::shared_ptr<const RawStorage>> owner_less;
        if (owner_less(lhs.owner, rhs.owner)) {
            return true;
        }

        if (owner_less(rhs.owner, lhs.owner)) {
            return false;
        }

        const auto lhs_address = reinterpret_cast<uintptr_t>(lhs.data);
        const auto rhs_address = reinterpret_cast<uintptr_t>(rhs.data);
        if (lhs_address != rhs_address) {
            return lhs_address < rhs_address;
        }

        if (lhs.bytes != rhs.bytes) {
            return lhs.bytes < rhs.bytes;
        }

        if (lhs.bfloat != rhs.bfloat) {
            return lhs.bfloat < rhs.bfloat;
        }
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
    if (weights.lm_head.has_value()) {
        views.push_back({&*weights.lm_head, "lm_head"});
    }

    for (size_t i = 0; i < weights.layers.size(); ++i) {
        auto& [norm, attn, mlp] = weights.layers[i];
        for (const auto& [view, name]: {
                     std::pair{&norm.input_rmsnorm, "input_rmsnorm"},
                     std::pair{&norm.post_attn_rmsnorm, "post_attn_rmsnorm"},
                     std::pair{&attn.q_proj, "q_proj"},
                     std::pair{&attn.k_proj, "k_proj"},
                     std::pair{&attn.v_proj, "v_proj"},
                     std::pair{&attn.o_proj, "o_proj"},
                     std::pair{&mlp.gate_proj, "gate_proj"},
                     std::pair{&mlp.up_proj, "up_proj"},
                     std::pair{&mlp.down_proj, "down_proj"}}) {
            views.push_back({view, "layer " + std::to_string(i) + " " + name});
        }
    }
    return views;
}

Status ValidateSource(const RawWeightView& view) {
    if (!view.IsValid()) {
        return Status::InvalidArgument(
                "FP32 weight conversion requires a valid source weight");
    }

    // Matching-code dtype equality calls lanes(); guard its scalar precondition
    // before equality or ValidateRawWeightView can inspect an encoded vector.
    if (view.dtype.raw_lanes() != 1 ||
        (view.dtype != DataType::Float32() && view.dtype != DataType::Float(16) &&
         view.dtype != DataType::BFloat(16))) {
        return Status::Unimplemented(
                "FP32 weight conversion requires scalar Float32, "
                "Float16, or BFloat16 weights");
    }

    AM_RETURN_IF_ERROR(ValidateRawWeightView(view));
    if (!view.is_contiguous) {
        return Status::InvalidArgument(
                "FP32 weight conversion requires contiguous source weights");
    }
    return Status::Ok();
}

float WidenToFp32(uint16_t bits, bool bfloat) {
    // Dtype widening preserves signaling NaNs. Quiet them in the source format
    // before converting to satisfy the weight conversion contract.
    if (bfloat) {
        if ((bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0) {
            bits |= 0x0040U;
        }
        return BFloat16{bits, BFloat16::from_bits()};
    }

    if ((bits & 0x7c00U) == 0x7c00U && (bits & 0x03ffU) != 0) {
        bits |= 0x0200U;
    }
    return Half{bits, Half::from_bits()};
}

Status AddBytes(size_t amount, size_t& total) {
    if (CheckOverflowAdd(total, amount, &total)) {
        return Status::Overflow(
                "FP32 weight conversion payload size overflowed size_t");
    }
    return Status::Ok();
}

} // namespace

StatusOr<std::unique_ptr<LoadedModel>> ConvertRawModelWeightsToFp32(
        const LoadedModel& model,
        Allocator& allocator,
        const RawWeightFp32ConversionOptions& options) try {
    if (!allocator.device().is_cpu()) {
        return Status::InvalidArgument(
                "FP32 weight conversion requires a CPU allocator");
    }

    auto converted = model.GetResolvedWeights();
    std::map<SourceKey, size_t, SourceLess> conversion_indices;
    std::vector<Conversion> conversions;
    RawWeightFp32ConversionUsage usage;
    for (const auto& [view, location]: CollectViews(converted)) {
        if (const auto status = ValidateSource(*view); !status.ok()) {
            return status.WithMessage("FP32 conversion weight " + location +
                                      ": " + status.message());
        }

        if (view->dtype == DataType::Float32()) {
            continue;
        }

        size_t target_bytes = 0;
        if (CheckOverflowMul(view->bytes / sizeof(uint16_t), sizeof(float),
                             &target_bytes)) {
            return Status::Overflow(
                    "FP32 converted weight byte size overflowed size_t");
        }

        SourceKey key{view->storage, view->data, view->bytes,
                      view->dtype == DataType::BFloat(16), view->shape};
        auto [entry, inserted] = conversion_indices.try_emplace(
                std::move(key), conversions.size());
        if (inserted) {
            AM_RETURN_IF_ERROR(AddBytes(view->bytes, usage.converted_source_bytes));
            AM_RETURN_IF_ERROR(AddBytes(target_bytes, usage.converted_weight_bytes));
            conversions.push_back({view->data,
                                   view->bytes,
                                   target_bytes,
                                   view->dtype == DataType::BFloat(16),
                                   {}});
        }
        conversions[entry->second].targets.push_back(view);
    }

    AM_ASSIGN_OR_RETURN(usage.source_weights, ComputeRawWeightStorageUsage(model.GetResolvedWeights()));
    usage.peak_owned_weight_bytes = usage.source_weights.owned_bytes;
    AM_RETURN_IF_ERROR(AddBytes(usage.converted_weight_bytes, usage.peak_owned_weight_bytes));
    usage.peak_owned_weight_bytes_complete = usage.source_weights.backing_usage_complete;

    // Converted roles no longer retain their source owners. Compute usage for
    // only F32 roles to count backing that survives alongside the new payload.
    ResolvedModelWeights retained = converted;
    for (const auto& [view, location]: CollectViews(retained)) {
        if (view->dtype != DataType::Float32()) {
            *view = {};
        }
    }

    AM_ASSIGN_OR_RETURN(auto retained_usage, ComputeRawWeightStorageUsage(retained));
    size_t final_owned_bytes = retained_usage.owned_bytes;
    AM_RETURN_IF_ERROR(AddBytes(usage.converted_weight_bytes, final_owned_bytes));
    if (options.max_owned_weight_bytes != 0) {
        if (!retained_usage.backing_usage_complete) {
            return Status::FailedPrecondition(
                    "Finite converted owned-weight cap requires complete "
                    "retained Float32 backing metadata");
        }

        if (final_owned_bytes > options.max_owned_weight_bytes) {
            return Status::ResourceExhausted(
                    "Converted owned-weight payload requires " +
                    std::to_string(final_owned_bytes) + " bytes; limit " +
                    std::to_string(options.max_owned_weight_bytes));
        }
    }

    for (const auto& conversion: conversions) {
        AM_ASSIGN_OR_RETURN(auto buffer,
                            allocator.TryAllocate(
                                    {conversion.target_bytes, 64}));
        auto* dest = static_cast<std::byte*>(buffer.mutable_data());
        for (size_t src_offset = 0, target_offset = 0;
             src_offset < conversion.source_bytes;
             src_offset += sizeof(uint16_t), target_offset += sizeof(float)) {
            uint16_t bits;
            std::memcpy(&bits, conversion.source + src_offset, sizeof(bits));
            const float value = WidenToFp32(bits, conversion.bfloat);
            // memcpy implicitly creates float objects in the raw allocation.
            std::memcpy(dest + target_offset, &value, sizeof(value));
        }

        auto owner = std::make_shared<ConvertedRawStorage>(std::move(buffer));
        for (auto* target: conversion.targets) {
            target->data = dest;
            target->bytes = conversion.target_bytes;
            target->dtype = DataType::Float32();
            target->storage = owner;
            AM_RETURN_IF_ERROR(ValidateRawWeightView(*target));
        }
    }
    AM_ASSIGN_OR_RETURN(usage.converted_weights, ComputeRawWeightStorageUsage(converted));
    return std::make_unique<LoadedModel>(model.GetConfig(), std::move(converted), usage);
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
