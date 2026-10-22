#include "aethermind/inference/model_execution_policy.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/model/loaded_model.h"
#include "aethermind/model/weight/weight_binding_resolver.h"

#include <array>
#include <cstdint>
#include <new>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

namespace aethermind {
namespace {

struct RequiredRole {
    TransformerWeightRole role;
    const char* name;
};

constexpr std::array kGlobalRoles{
        RequiredRole{TransformerWeightRole::kTokenEmbedding, "TokenEmbedding"},
        RequiredRole{TransformerWeightRole::kFinalNorm, "FinalNorm"},
        RequiredRole{TransformerWeightRole::kLmHead, "LmHead"},
};

constexpr std::array kLayerRoles{
        RequiredRole{TransformerWeightRole::kInputNorm, "InputNorm"},
        RequiredRole{TransformerWeightRole::kPostAttentionNorm, "PostAttentionNorm"},
        RequiredRole{TransformerWeightRole::kAttentionQ, "AttentionQ"},
        RequiredRole{TransformerWeightRole::kAttentionK, "AttentionK"},
        RequiredRole{TransformerWeightRole::kAttentionV, "AttentionV"},
        RequiredRole{TransformerWeightRole::kAttentionO, "AttentionO"},
        RequiredRole{TransformerWeightRole::kMlpGate, "MlpGate"},
        RequiredRole{TransformerWeightRole::kMlpUp, "MlpUp"},
        RequiredRole{TransformerWeightRole::kMlpDown, "MlpDown"},
};

std::string WeightLocation(const RequiredRole& role, std::optional<uint32_t> layer) {
    std::string location = role.name;
    if (layer.has_value()) {
        location += " at layer " + std::to_string(*layer);
    }
    return location;
}

Status ValidateRoleDType(const ResolvedModelWeights& weights,
                         const RequiredRole& role,
                         std::optional<uint32_t> layer) {
    const RawWeightView* weight = ResolveWeightBinding(
            MakeTransformerWeightBinding(layer, role.role), weights);
    if (weight == nullptr || !weight->IsValid()) {
        return Status::InvalidArgument("Required weight " + WeightLocation(role, layer) +
                                       " is missing or invalid");
    }
    // IsFloat32() does not check lanes; kernels require scalar elements.
    if (weight->dtype != DataType::Float32()) {
        return Status::Unimplemented("Weight " + WeightLocation(role, layer) +
                                     " has dtype " + ToString(weight->dtype) +
                                     "; required Float32");
    }
    return Status::Ok();
}

} // namespace

Status ValidateFp32ExecutionInputs(const LoadedModel& model) try {
    const auto& config = model.GetConfig();
    const auto hint = config.weight_dtype_hint;
    const bool automatic_hint = hint.IsUndefined() &&
                                (config.weight_dtype_hint_name.empty() ||
                                 config.weight_dtype_hint_name == "auto");
    if (hint != DataType::Float32() && !automatic_hint) {
        return Status::Unimplemented(
                "Model weight dtype hint '" + config.weight_dtype_hint_name +
                "' has dtype " + ToString(hint) + "; required Float32");
    }

    const auto& weights = model.GetResolvedWeights();
    AM_ASSIGN_OR_RETURN(const size_t num_layers,
                        PositiveSize(config.num_hidden_layers, "num_hidden_layers"));
    if (weights.layers.size() != num_layers) {
        return Status::InvalidArgument("Resolved layer count does not match num_hidden_layers");
    }
    if (!std::in_range<uint32_t>(num_layers - 1)) {
        return Status::OutOfRange("Decoder layer index is not representable as uint32_t");
    }
    for (const auto& role: kGlobalRoles) {
        AM_RETURN_IF_ERROR(ValidateRoleDType(weights, role, std::nullopt));
    }
    for (size_t layer = 0; layer < num_layers; ++layer) {
        for (const auto& role: kLayerRoles) {
            AM_RETURN_IF_ERROR(ValidateRoleDType(weights, role, static_cast<uint32_t>(layer)));
        }
    }
    return Status::Ok();
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

StatusOr<KVCacheRuntimeOptions> MakeFp32KVCacheOptions(
        const HfModelConfig& config, size_t context_tokens) try {
    AM_ASSIGN_OR_RETURN(const size_t num_layers, PositiveSize(config.num_hidden_layers, "num_hidden_layers"));
    AM_ASSIGN_OR_RETURN(const size_t num_kv_heads, PositiveSize(config.num_key_value_heads, "num_key_value_heads"));
    AM_ASSIGN_OR_RETURN(const size_t model_context, PositiveSize(config.max_position_embeddings, "max_position_embeddings"));
    if (context_tokens == 0) {
        return Status::InvalidArgument("context_tokens must be positive");
    }
    if (context_tokens > model_context) {
        return Status::OutOfRange("context_tokens exceeds max_position_embeddings");
    }

    int64_t logical_head_dim = config.head_dim;
    if (logical_head_dim == 0) {
        // Only the loader's absent head_dim representation uses this fallback.
        if (config.hidden_size <= 0 || config.num_attention_heads <= 0 ||
            config.hidden_size % config.num_attention_heads != 0) {
            return Status::InvalidArgument("head_dim requires positive, divisible hidden_size and num_attention_heads");
        }
        logical_head_dim = config.hidden_size / config.num_attention_heads;
    }
    AM_ASSIGN_OR_RETURN(const size_t head_dim, PositiveSize(logical_head_dim, "head_dim"));

    return KVCacheRuntimeOptions{.enable_manager = true,
                                 .num_layers = num_layers,
                                 .num_kv_heads = num_kv_heads,
                                 .max_tokens = context_tokens,
                                 .head_dim = head_dim,
                                 .kv_dtype = DataType::Float32(),
                                 .alignment = 64};
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

} // namespace aethermind
