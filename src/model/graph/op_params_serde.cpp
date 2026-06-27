#include "aethermind/model/graph/op_params_serde.h"
#include "utils/variant_utils.h"

#include <charconv>
#include <ostream>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <variant>

namespace aethermind {
namespace {

using FieldMap = std::unordered_map<std::string, std::string>;

StatusOr<int64_t> ParseInt64(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing integer field");
    }

    int64_t value = 0;
    const std::string& text = it->second;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid integer field");
    }
    return value;
}

StatusOr<float> ParseFloat(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing float field");
    }

    float value = 0.0F;
    const std::string& text = it->second;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid float field");
    }
    return value;
}

StatusOr<double> ParseDouble(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing double field");
    }

    double value = 0.0;
    const std::string& text = it->second;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid double field");
    }
    return value;
}

StatusOr<bool> ParseBool(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing bool field");
    }

    if (it->second == "true") {
        return true;
    }

    if (it->second == "false") {
        return false;
    }
    return Status::InvalidArgument("ParseOpParams: invalid bool field");
}

FieldMap ParseFields(std::istringstream& input) {
    FieldMap fields;
    std::string token;
    while (input >> token) {
        const size_t pos = token.find('=');
        if (pos == std::string::npos) {
            fields.emplace(std::move(token), std::string{});
            continue;
        }
        fields.emplace(token.substr(0, pos), token.substr(pos + 1));
    }
    return fields;
}

StatusOr<HfRopeScalingType> ParseRopeScalingField(const FieldMap& fields) {
    const auto it = fields.find("scaling_type");
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing scaling_type field");
    }

    const HfRopeScalingType scaling_type = ParseRopeScalingType(it->second);
    if (scaling_type == HfRopeScalingType::kUnknown && it->second != "unknown") {
        return Status::InvalidArgument("ParseOpParams: invalid scaling_type field");
    }
    return scaling_type;
}

StatusOr<std::optional<double>> ParseOptionalDouble(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing optional double field");
    }

    if (it->second == "none") {
        return std::optional<double>{};
    }
    double value = 0.0;
    const std::string& text = it->second;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid optional double field");
    }
    return std::optional<double>{value};
}

Status EnsureNoExtraFields(const FieldMap& fields, size_t expected_count) {
    if (fields.size() != expected_count) {
        return Status::InvalidArgument("ParseOpParams: unexpected field count");
    }
    return Status::Ok();
}

}// namespace

const char* OpParamsKindName(const OpParams& params) noexcept {
    auto visitor = overloaded{
            [](const std::monostate&) noexcept { return "monostate"; },
            [](const EmbeddingParams&) noexcept { return "Embedding"; },
            [](const RmsNormParams&) noexcept { return "RmsNorm"; },
            [](const LinearParams&) noexcept { return "Linear"; },
            [](const RoPEParams&) noexcept { return "RoPE"; },
            [](const MatMulParams&) noexcept { return "MatMul"; },
            [](const SoftmaxParams&) noexcept { return "Softmax"; },
            [](const AddParams&) noexcept { return "Add"; },
            [](const SiluMulParams&) noexcept { return "SiluMul"; },
            [](const KVCacheUpdateParams&) noexcept { return "KVCacheUpdate"; },
            [](const AttentionParams&) noexcept { return "Attention"; },
            [](const ArgmaxParams&) noexcept { return "Argmax"; },
    };
    return std::visit(visitor, params);
}

Status SerializeOpParams(const OpParams& params, std::ostream& os) {
    auto visitor = overloaded{
            [&](const std::monostate&) { os << "monostate"; },
            [&](const EmbeddingParams&) { os << "Embedding"; },
            [&](const RmsNormParams& p) { os << "RmsNorm eps=" << p.eps; },
            [&](const LinearParams&) { os << "Linear"; },
            [&](const RoPEParams& p) {
                os << "RoPE head_dim=" << p.head_dim
                   << " num_attention_heads=" << p.num_attention_heads
                   << " num_key_value_heads=" << p.num_key_value_heads
                   << " max_position_embeddings=" << p.max_position_embeddings
                   << " theta=" << p.theta
                   << " scaling_factor=";
                if (p.scaling_factor.has_value()) {
                    os << *p.scaling_factor;
                } else {
                    os << "none";
                }
                os << " scaling_type=" << ToString(p.scaling_type);
            },
            [&](const MatMulParams& p) {
                os << "MatMul transpose_rhs=" << (p.transpose_rhs ? "true" : "false");
            },
            [&](const SoftmaxParams& p) { os << "Softmax axis=" << p.axis; },
            [&](const AddParams&) { os << "Add"; },
            [&](const SiluMulParams&) { os << "SiluMul"; },
            [&](const KVCacheUpdateParams&) { os << "KVCacheUpdate"; },
            [&](const AttentionParams& p) {
                os << "Attention num_attention_heads=" << p.num_attention_heads
                   << " num_key_value_heads=" << p.num_key_value_heads
                   << " head_dim=" << p.head_dim;
            },
            [&](const ArgmaxParams& p) { os << "Argmax axis=" << p.axis; },
    };
    std::visit(visitor, params);
    return Status::Ok();
}

StatusOr<OpParams> ParseOpParams(std::string_view text) {
    std::istringstream input(std::string{text});
    std::string kind;
    if (!(input >> kind)) {
        return Status::InvalidArgument("ParseOpParams: empty input");
    }
    FieldMap fields = ParseFields(input);

    if (kind == "monostate") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{std::monostate{}};
    }

    if (kind == "Embedding") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{EmbeddingParams{}};
    }

    if (kind == "RmsNorm") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        StatusOr<float> eps = ParseFloat(fields, "eps");
        AM_RETURN_IF_ERROR(eps.status());
        return OpParams{RmsNormParams{.eps = *eps}};
    }

    if (kind == "Linear") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{LinearParams{}};
    }

    if (kind == "RoPE") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 7));
        StatusOr<int64_t> head_dim = ParseInt64(fields, "head_dim");
        AM_RETURN_IF_ERROR(head_dim.status());
        StatusOr<int64_t> num_attention_heads = ParseInt64(fields, "num_attention_heads");
        AM_RETURN_IF_ERROR(num_attention_heads.status());
        StatusOr<int64_t> num_key_value_heads = ParseInt64(fields, "num_key_value_heads");
        AM_RETURN_IF_ERROR(num_key_value_heads.status());
        StatusOr<int64_t> max_position_embeddings = ParseInt64(fields, "max_position_embeddings");
        AM_RETURN_IF_ERROR(max_position_embeddings.status());
        StatusOr<double> theta = ParseDouble(fields, "theta");
        AM_RETURN_IF_ERROR(theta.status());
        StatusOr<std::optional<double>> scaling_factor = ParseOptionalDouble(fields, "scaling_factor");
        AM_RETURN_IF_ERROR(scaling_factor.status());
        StatusOr<HfRopeScalingType> scaling_type = ParseRopeScalingField(fields);
        AM_RETURN_IF_ERROR(scaling_type.status());
        return OpParams{RoPEParams{.head_dim = *head_dim,
                                   .num_attention_heads = *num_attention_heads,
                                   .num_key_value_heads = *num_key_value_heads,
                                   .max_position_embeddings = *max_position_embeddings,
                                   .theta = *theta,
                                   .scaling_factor = *scaling_factor,
                                   .scaling_type = *scaling_type}};
    }

    if (kind == "MatMul") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        StatusOr<bool> transpose_rhs = ParseBool(fields, "transpose_rhs");
        AM_RETURN_IF_ERROR(transpose_rhs.status());
        return OpParams{MatMulParams{.transpose_rhs = *transpose_rhs}};
    }

    if (kind == "Softmax") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        StatusOr<int64_t> axis = ParseInt64(fields, "axis");
        AM_RETURN_IF_ERROR(axis.status());
        return OpParams{SoftmaxParams{.axis = *axis}};
    }

    if (kind == "Add") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{AddParams{}};
    }

    if (kind == "SiluMul") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{SiluMulParams{}};
    }

    if (kind == "KVCacheUpdate") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{KVCacheUpdateParams{}};
    }

    if (kind == "Attention") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 3));
        StatusOr<int64_t> num_attention_heads = ParseInt64(fields, "num_attention_heads");
        AM_RETURN_IF_ERROR(num_attention_heads.status());
        StatusOr<int64_t> num_key_value_heads = ParseInt64(fields, "num_key_value_heads");
        AM_RETURN_IF_ERROR(num_key_value_heads.status());
        StatusOr<int64_t> head_dim = ParseInt64(fields, "head_dim");
        AM_RETURN_IF_ERROR(head_dim.status());
        return OpParams{AttentionParams{.num_attention_heads = *num_attention_heads,
                                        .num_key_value_heads = *num_key_value_heads,
                                        .head_dim = *head_dim}};
    }

    if (kind == "Argmax") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        StatusOr<int64_t> axis = ParseInt64(fields, "axis");
        AM_RETURN_IF_ERROR(axis.status());
        return OpParams{ArgmaxParams{.axis = *axis}};
    }

    return Status::InvalidArgument("ParseOpParams: unknown parameter kind");
}

}// namespace aethermind
