#include "aethermind/model/graph/op_params_serde.h"
#include "utils/variant_utils.h"

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

StatusOr<FieldMap> ParseFields(std::istringstream& input) {
    FieldMap fields;
    std::string token;
    while (input >> token) {
        const size_t pos = token.find('=');
        std::string name = (pos == std::string::npos) ? std::move(token) : token.substr(0, pos);
        std::string value = (pos == std::string::npos) ? std::string{} : token.substr(pos + 1);
        // Reject duplicate field names — every op requires each field at most
        // once. unordered_map::emplace would silently drop the second occurrence
        // and mask the malformed input.
        if (fields.find(name) != fields.end()) {
            return Status::InvalidArgument("ParseOpParams: duplicate field '" + name + "'");
        }
        fields.emplace(std::move(name), std::move(value));
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

// Parses a single Reshape target-dimension token into a ReshapeDim variant.
// Tokens are: a non-negative decimal literal (`32`), `@N` with N an unsigned
// decimal that fits uint32_t (`@0`, `@4294967295`), or `*` (infer).
// Returns InvalidArgument for empty tokens, signed/negative literals,
// `@` without digits, axis overflow, or any unknown syntax.
StatusOr<ReshapeDim> ParseReshapeDimToken(std::string_view token) {
    if (token.empty()) {
        return Status::InvalidArgument("ParseOpParams: empty Reshape dim token");
    }

    if (token.size() == 1 && token[0] == '*') {
        return ReshapeDim{ReshapeInferDim{}};
    }

    if (token[0] == '@') {
        const std::string_view digits = token.substr(1);
        if (digits.empty()) {
            return Status::InvalidArgument(
                    "ParseOpParams: Reshape axis reference without digits");
        }
        // Reject leading zeros except for the canonical "0" form, and reject
        // any non-digit character to keep numeric spelling strict.
        for (char c: digits) {
            if (c < '0' || c > '9') {
                return Status::InvalidArgument(
                        "ParseOpParams: non-digit in Reshape axis reference");
            }
        }
        // Parse as uint64_t first to detect overflow beyond uint32_t range.
        uint64_t axis_value = 0;
        const auto result = std::from_chars(
                digits.data(), digits.data() + digits.size(), axis_value);
        if (result.ec != std::errc{} || result.ptr != digits.data() + digits.size()) {
            return Status::InvalidArgument("ParseOpParams: invalid Reshape axis reference");
        }

        if (axis_value > std::numeric_limits<uint32_t>::max()) {
            return Status::InvalidArgument("ParseOpParams: Reshape axis reference overflow");
        }
        return ReshapeDim{ReshapeInputDim{.axis = static_cast<uint32_t>(axis_value)}};
    }

    // Literal: must be a non-negative decimal. Reject any leading sign or
    // non-digit character so signed/negative forms fail explicitly.
    for (char c: token) {
        if (c < '0' || c > '9') {
            return Status::InvalidArgument("ParseOpParams: invalid Reshape literal dim");
        }
    }
    int64_t literal_value = 0;
    const auto result = std::from_chars(token.data(), token.data() + token.size(), literal_value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid Reshape literal dim");
    }
    return ReshapeDim{ReshapeLiteralDim{.value = literal_value}};
}

// Parses the canonical `shape=[...]` field value into a vector of ReshapeDim.
// Rejects missing brackets, mismatched brackets, leading/trailing/consecutive
// commas, empty tokens, and any malformed dim token. An empty interior `[]`
// is valid and yields an empty vector (rank zero).
StatusOr<std::vector<ReshapeDim>> ParseReshapeShape(std::string_view value) {
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
        return Status::InvalidArgument("ParseOpParams: Reshape shape must be bracketed");
    }

    const std::string_view interior = value.substr(1, value.size() - 2);
    std::vector<ReshapeDim> shape;
    if (interior.empty()) {
        return shape;
    }

    shape.reserve(std::count(interior.begin(), interior.end(), ',') + 1);
    std::string_view::size_type start = 0;
    while (true) {
        const auto end = interior.find(',', start);
        const std::string_view token = interior.substr(
                start,
                end == std::string_view::npos ? std::string_view::npos : end - start);
        if (token.empty()) {
            return Status::InvalidArgument("ParseOpParams: empty Reshape dim token");
        }

        StatusOr<ReshapeDim> dim = ParseReshapeDimToken(token);
        AM_RETURN_IF_ERROR(dim.status());
        shape.push_back(*std::move(dim));
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return shape;
}

}// namespace

// Serializes a Reshape target_shape to its canonical textual form, e.g.
// `[@0,@1,32,*]`. Tokens are joined with commas and no interior whitespace.
// Literal dims emit their decimal value, input-axis references emit `@N`
// with N an unsigned decimal, and the infer marker emits `*`. An empty
// vector emits `[]` (rank zero).
// Exposed in op_params_serde.h so DumpOpParams in graph_dump.cpp shares the
// canonical spelling instead of duplicating the format.
void SerializeReshapeShape(const std::vector<ReshapeDim>& target_shape, std::ostream& os) {
    os << '[';
    bool first = true;
    for (const ReshapeDim& dim: target_shape) {
        if (!first) {
            os << ',';
        }

        first = false;
        std::visit(overloaded{
                           [&](const ReshapeLiteralDim& d) { os << d.value; },
                           [&](const ReshapeInputDim& d) { os << '@' << d.axis; },
                           [&](const ReshapeInferDim) { os << '*'; },
                   },
                   dim);
    }
    os << ']';
}

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
            [](const SiluParams&) noexcept { return "Silu"; },
            [](const SiluMulParams&) noexcept { return "SiluMul"; },
            [](const ElementwiseMulParams&) noexcept { return "ElementwiseMul"; },
            [](const KVCacheUpdateParams&) noexcept { return "KVCacheUpdate"; },
            [](const AttentionParams&) noexcept { return "Attention"; },
            [](const ArgmaxParams&) noexcept { return "Argmax"; },
            [](const ReshapeParams&) noexcept { return "Reshape"; },
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
            [&](const SiluParams&) { os << "Silu"; },
            [&](const SiluMulParams&) { os << "SiluMul"; },
            [&](const ElementwiseMulParams&) { os << "ElementwiseMul"; },
            [&](const KVCacheUpdateParams&) { os << "KVCacheUpdate"; },
            [&](const AttentionParams& p) {
                os << "Attention num_attention_heads=" << p.num_attention_heads
                   << " num_key_value_heads=" << p.num_key_value_heads
                   << " head_dim=" << p.head_dim;
            },
            [&](const ArgmaxParams& p) { os << "Argmax axis=" << p.axis; },
            [&](const ReshapeParams& p) {
                os << "Reshape shape=";
                SerializeReshapeShape(p.target_shape, os);
            },
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
    StatusOr<FieldMap> fields_or = ParseFields(input);
    AM_RETURN_IF_ERROR(fields_or.status());
    const FieldMap& fields = *fields_or;

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

    if (kind == "Silu") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{SiluParams{}};
    }

    if (kind == "SiluMul") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{SiluMulParams{}};
    }

    if (kind == "ElementwiseMul") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{ElementwiseMulParams{}};
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

    if (kind == "Reshape") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        const auto it = fields.find("shape");
        if (it == fields.end()) {
            return Status::InvalidArgument("ParseOpParams: missing shape field");
        }
        StatusOr<std::vector<ReshapeDim>> target_shape = ParseReshapeShape(it->second);
        AM_RETURN_IF_ERROR(target_shape.status());
        return OpParams{ReshapeParams{.target_shape = std::move(*target_shape)}};
    }

    return Status::InvalidArgument("ParseOpParams: unknown parameter kind");
}

}// namespace aethermind
