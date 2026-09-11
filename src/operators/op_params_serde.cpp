#include "aethermind/operators/op_params_serde.h"
#include "utils/parse_number.h"
#include "utils/variant_utils.h"

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

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
    if (!utils::ParseFloat(it->second, value)) {
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
    if (!utils::ParseDouble(it->second, value)) {
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
        if (fields.contains(name)) {
            return Status::InvalidArgument("ParseOpParams: duplicate field '" + name + "'");
        }
        fields.emplace(std::move(name), std::move(value));
    }
    return fields;
}

StatusOr<RoPEAlgorithm> ParseRoPEAlgorithmField(const FieldMap& fields) {
    const auto it = fields.find("algorithm");
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing RoPE algorithm field");
    }
    if (it->second == "standard") return RoPEAlgorithm::kStandard;
    if (it->second == "linear") return RoPEAlgorithm::kLinear;
    if (it->second == "dynamic_ntk") return RoPEAlgorithm::kDynamicNtk;
    if (it->second == "yarn") return RoPEAlgorithm::kYarn;
    if (it->second == "llama3") return RoPEAlgorithm::kLlama3;
    if (it->second == "longrope") return RoPEAlgorithm::kLongRope;
    return Status::InvalidArgument("ParseOpParams: invalid RoPE algorithm field");
}

StatusOr<RoPEPairing> ParseRoPEPairingField(const FieldMap& fields) {
    const auto it = fields.find("pairing");
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing RoPE pairing field");
    }
    if (it->second == "split_half") return RoPEPairing::kSplitHalf;
    if (it->second == "interleaved") return RoPEPairing::kInterleaved;
    return Status::InvalidArgument("ParseOpParams: invalid RoPE pairing field");
}

StatusOr<RoPEAlgorithm> ParseLegacyRopeScalingField(const FieldMap& fields) {
    const auto it = fields.find("scaling_type");
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing scaling_type field");
    }

    const std::string_view value = it->second;
    if (value == "none") {
        return RoPEAlgorithm::kStandard;
    }
    if (value == "linear") {
        return RoPEAlgorithm::kLinear;
    }
    return Status::InvalidArgument("ParseOpParams: invalid scaling_type field");
}

StatusOr<std::vector<double>> ParseDoubleList(const FieldMap& fields, std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end() || it->second.empty()) {
        return Status::InvalidArgument("ParseOpParams: missing double list field");
    }
    std::vector<double> values;
    std::string_view input = it->second;
    while (!input.empty()) {
        const size_t comma = input.find(',');
        const std::string_view token = input.substr(0, comma);
        double value = 0.0;
        if (token.empty() || !utils::ParseDouble(token, value)) {
            return Status::InvalidArgument("ParseOpParams: invalid double list field");
        }
        values.push_back(value);
        if (comma == std::string_view::npos) break;
        input.remove_prefix(comma + 1);
    }
    return values;
}

void SerializeDoubleList(const std::vector<double>& values, std::ostream& os) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) os << ',';
        os << values[i];
    }
}

StatusOr<std::optional<double>> ParseOptionalDouble(const FieldMap& fields,
                                                    std::string_view name) {
    const auto it = fields.find(std::string(name));
    if (it == fields.end()) {
        return Status::InvalidArgument("ParseOpParams: missing optional double field");
    }

    if (it->second == "none") {
        return std::optional<double>{};
    }
    double value = 0.0;
    if (!utils::ParseDouble(it->second, value)) {
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
            return Status::InvalidArgument(
                    "ParseOpParams: Reshape axis reference overflow");
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
    const auto result = std::from_chars(token.data(),
                                        token.data() + token.size(),
                                        literal_value);
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

// Parses a single Permute axis token. Tokens must be a canonical unsigned
// decimal that fits uint32_t: exactly "0" or a non-zero-leading sequence of
// digits. Rejects empty tokens, signs, non-digits, leading zeros (except
// "0" itself), and values above uint32_t::max().
StatusOr<uint32_t> ParsePermuteAxisToken(std::string_view token) {
    if (token.empty()) {
        return Status::InvalidArgument("ParseOpParams: empty Permute axis token");
    }

    for (char c: token) {
        if (c < '0' || c > '9') {
            return Status::InvalidArgument(
                    "ParseOpParams: non-digit in Permute axis");
        }
    }

    // Reject non-canonical leading zeros except for the single digit "0".
    if (token.size() > 1 && token[0] == '0') {
        return Status::InvalidArgument(
                "ParseOpParams: non-canonical leading zero in Permute axis");
    }

    // Parse as uint64_t first to detect overflow beyond uint32_t range.
    uint64_t axis_value = 0;
    const auto result = std::from_chars(
            token.data(), token.data() + token.size(), axis_value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) {
        return Status::InvalidArgument("ParseOpParams: invalid Permute axis");
    }

    if (axis_value > std::numeric_limits<uint32_t>::max()) {
        return Status::InvalidArgument("ParseOpParams: Permute axis overflow");
    }
    return static_cast<uint32_t>(axis_value);
}

// Parses the canonical `permutation=[...]` field value into a vector of
// uint32_t axis indexes. Rejects missing brackets, mismatched brackets,
// leading/trailing/consecutive commas, empty tokens, signed/negative
// values, non-digits, non-canonical leading zeros, and values above
// uint32_t::max(). An empty interior `[]` is valid and yields an empty
// vector (rank zero). Repeated axes (e.g. `[0,0]`) parse successfully;
// bijection semantics are owned by InferPermute, not serde.
StatusOr<std::vector<uint32_t>> ParsePermutation(std::string_view value) {
    if (value.size() < 2 || value.front() != '[' || value.back() != ']') {
        return Status::InvalidArgument(
                "ParseOpParams: Permute permutation must be bracketed");
    }

    const std::string_view interior = value.substr(1, value.size() - 2);
    std::vector<uint32_t> permutation;
    if (interior.empty()) {
        return permutation;
    }

    permutation.reserve(std::count(interior.begin(), interior.end(), ',') + 1);
    std::string_view::size_type start = 0;
    while (true) {
        const auto end = interior.find(',', start);
        const std::string_view token = interior.substr(
                start,
                end == std::string_view::npos ? std::string_view::npos : end - start);
        if (token.empty()) {
            return Status::InvalidArgument(
                    "ParseOpParams: empty Permute axis token");
        }

        StatusOr<uint32_t> axis = ParsePermuteAxisToken(token);
        AM_RETURN_IF_ERROR(axis.status());
        permutation.push_back(*axis);
        if (end == std::string_view::npos) {
            break;
        }
        start = end + 1;
    }
    return permutation;
}

} // namespace

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
        auto visitor = overloaded{
                [&](const ReshapeLiteralDim& d) { os << d.value; },
                [&](const ReshapeInputDim& d) { os << '@' << d.axis; },
                [&](const ReshapeInferDim) { os << '*'; },
        };
        std::visit(visitor, dim);
    }
    os << ']';
}

// Serializes a Permute permutation to its canonical textual form, e.g.
// `[2,0,1]`. Tokens are unsigned decimals joined with commas and no interior
// whitespace. An empty vector emits `[]` (rank zero).
void SerializePermutation(const std::vector<uint32_t>& permutation, std::ostream& os) {
    os << '[';
    bool first = true;
    for (uint32_t axis: permutation) {
        if (!first) {
            os << ',';
        }
        first = false;
        os << axis;
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
            [](const PermuteParams&) noexcept { return "Permute"; },
            [](const ReorderParams&) noexcept { return "Reorder"; },
            [](const QkvLinearParams&) noexcept { return "QkvLinear"; },
            [](const GateUpLinearParams&) noexcept { return "GateUpLinear"; },
            [](const AddRmsNormParams&) noexcept { return "AddRmsNorm"; },
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
                os << "RoPE version=2 head_dim=" << p.head_dim
                   << " rotary_dim=" << EffectiveRoPERotaryDim(p)
                   << " num_q_heads=" << p.num_q_heads
                   << " num_kv_heads=" << p.num_kv_heads
                   << " max_pos_embeddings=" << p.max_pos_embeddings
                   << " theta=" << p.theta
                   << " pairing=" << ToString(p.pairing)
                   << " algorithm=" << ToString(GetRoPEAlgorithm(p.algorithm));
                std::visit(
                        [&](const auto& algorithm) {
                            using T = std::decay_t<decltype(algorithm)>;
                            if constexpr (std::is_same_v<T, LinearRoPE>) {
                                os << " factor=" << algorithm.factor;
                            } else if constexpr (std::is_same_v<T, DynamicNtkRoPE>) {
                                os << " factor=" << algorithm.factor
                                   << " original_context_length=" << algorithm.original_context_length;
                            } else if constexpr (std::is_same_v<T, YarnRoPE>) {
                                os << " factor=" << algorithm.factor
                                   << " original_context_length=" << algorithm.original_context_length
                                   << " beta_fast=" << algorithm.beta_fast
                                   << " beta_slow=" << algorithm.beta_slow
                                   << " rotary_output_scale=" << algorithm.rotary_output_scale
                                   << " truncate_correction_range="
                                   << (algorithm.truncate_correction_range ? "true" : "false");
                            } else if constexpr (std::is_same_v<T, Llama3RoPE>) {
                                os << " factor=" << algorithm.factor
                                   << " low_frequency_factor=" << algorithm.low_frequency_factor
                                   << " high_frequency_factor=" << algorithm.high_frequency_factor
                                   << " original_context_length=" << algorithm.original_context_length;
                            } else if constexpr (std::is_same_v<T, LongRoPE>) {
                                os << " original_context_length=" << algorithm.original_context_length
                                   << " rotary_output_scale=" << algorithm.rotary_output_scale
                                   << " short_factors=";
                                SerializeDoubleList(algorithm.short_factors, os);
                                os << " long_factors=";
                                SerializeDoubleList(algorithm.long_factors, os);
                            }
                        },
                        p.algorithm);
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
                os << "Attention num_q_heads=" << p.num_q_heads
                   << " num_kv_heads=" << p.num_kv_heads
                   << " head_dim=" << p.head_dim;
            },
            [&](const ArgmaxParams& p) { os << "Argmax axis=" << p.axis; },
            [&](const ReshapeParams& p) {
                os << "Reshape shape=";
                SerializeReshapeShape(p.target_shape, os);
            },
            [&](const PermuteParams& p) {
                os << "Permute permutation=";
                SerializePermutation(p.permutation, os);
            },
            [&](const ReorderParams&) { os << "Reorder"; },
            [&](const QkvLinearParams& p) {
                os << "QkvLinear q_out_features=" << p.q_out_features
                   << " k_out_features=" << p.k_out_features
                   << " v_out_features=" << p.v_out_features
                   << " has_bias=" << (p.has_bias ? "true" : "false");
            },
            [&](const GateUpLinearParams& p) {
                os << "GateUpLinear gate_out_features=" << p.gate_out_features
                   << " up_out_features=" << p.up_out_features
                   << " has_bias=" << (p.has_bias ? "true" : "false");
            },
            [&](const AddRmsNormParams& p) {
                os << "AddRmsNorm eps=" << p.eps;
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
        const bool legacy = !fields.contains("version");
        if (legacy) {
            AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 7));
            StatusOr<int64_t> head_dim = ParseInt64(fields, "head_dim");
            AM_RETURN_IF_ERROR(head_dim.status());
            StatusOr<int64_t> num_q_heads = ParseInt64(fields, "num_q_heads");
            AM_RETURN_IF_ERROR(num_q_heads.status());
            StatusOr<int64_t> num_kv_heads = ParseInt64(fields, "num_kv_heads");
            AM_RETURN_IF_ERROR(num_kv_heads.status());
            StatusOr<int64_t> max_pos_embeddings = ParseInt64(fields, "max_pos_embeddings");
            AM_RETURN_IF_ERROR(max_pos_embeddings.status());
            StatusOr<double> theta = ParseDouble(fields, "theta");
            AM_RETURN_IF_ERROR(theta.status());
            StatusOr<std::optional<double>> scaling_factor = ParseOptionalDouble(fields, "scaling_factor");
            AM_RETURN_IF_ERROR(scaling_factor.status());
            StatusOr<RoPEAlgorithm> algorithm = ParseLegacyRopeScalingField(fields);
            AM_RETURN_IF_ERROR(algorithm.status());
            RoPEAlgorithmParams params = StandardRoPE{};
            if (*algorithm == RoPEAlgorithm::kLinear) {
                if (!scaling_factor->has_value()) {
                    return Status::InvalidArgument("ParseOpParams: linear legacy RoPE lacks scaling_factor");
                }
                params = LinearRoPE{.factor = **scaling_factor};
            } else if (scaling_factor->has_value()) {
                return Status::InvalidArgument("ParseOpParams: standard legacy RoPE has scaling_factor");
            }
            return OpParams{RoPEParams{.head_dim = *head_dim,
                                       .rotary_dim = *head_dim,
                                       .num_q_heads = *num_q_heads,
                                       .num_kv_heads = *num_kv_heads,
                                       .max_pos_embeddings = *max_pos_embeddings,
                                       .theta = *theta,
                                       .algorithm = std::move(params)}};
        }

        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, [&] {
            StatusOr<int64_t> version = ParseInt64(fields, "version");
            if (!version.ok() || *version != 2) return size_t{0};
            StatusOr<RoPEAlgorithm> algorithm = ParseRoPEAlgorithmField(fields);
            if (!algorithm.ok()) return size_t{0};
            switch (*algorithm) {
                case RoPEAlgorithm::kStandard:
                    return size_t{9};
                case RoPEAlgorithm::kLinear:
                    return size_t{10};
                case RoPEAlgorithm::kDynamicNtk:
                    return size_t{11};
                case RoPEAlgorithm::kYarn:
                    return size_t{15};
                case RoPEAlgorithm::kLlama3:
                    return size_t{13};
                case RoPEAlgorithm::kLongRope:
                    return size_t{13};
            }
            return size_t{0};
        }()));
        StatusOr<int64_t> version = ParseInt64(fields, "version");
        AM_RETURN_IF_ERROR(version.status());
        if (*version != 2) return Status::InvalidArgument("ParseOpParams: unsupported RoPE version");
        StatusOr<int64_t> head_dim = ParseInt64(fields, "head_dim");
        AM_RETURN_IF_ERROR(head_dim.status());
        StatusOr<int64_t> rotary_dim = ParseInt64(fields, "rotary_dim");
        AM_RETURN_IF_ERROR(rotary_dim.status());
        StatusOr<int64_t> num_q_heads = ParseInt64(fields, "num_q_heads");
        AM_RETURN_IF_ERROR(num_q_heads.status());
        StatusOr<int64_t> num_kv_heads = ParseInt64(fields, "num_kv_heads");
        AM_RETURN_IF_ERROR(num_kv_heads.status());
        StatusOr<int64_t> max_pos_embeddings = ParseInt64(fields, "max_pos_embeddings");
        AM_RETURN_IF_ERROR(max_pos_embeddings.status());
        StatusOr<double> theta = ParseDouble(fields, "theta");
        AM_RETURN_IF_ERROR(theta.status());
        StatusOr<RoPEPairing> pairing = ParseRoPEPairingField(fields);
        AM_RETURN_IF_ERROR(pairing.status());
        StatusOr<RoPEAlgorithm> algorithm = ParseRoPEAlgorithmField(fields);
        AM_RETURN_IF_ERROR(algorithm.status());
        RoPEAlgorithmParams algorithm_params = StandardRoPE{};
        switch (*algorithm) {
            case RoPEAlgorithm::kStandard:
                break;
            case RoPEAlgorithm::kLinear: {
                AM_ASSIGN_OR_RETURN(const double factor, ParseDouble(fields, "factor"));
                algorithm_params = LinearRoPE{.factor = factor};
                break;
            }
            case RoPEAlgorithm::kDynamicNtk: {
                AM_ASSIGN_OR_RETURN(const double factor, ParseDouble(fields, "factor"));
                AM_ASSIGN_OR_RETURN(const int64_t original,
                                    ParseInt64(fields, "original_context_length"));
                algorithm_params = DynamicNtkRoPE{.factor = factor,
                                                  .original_context_length = original};
                break;
            }
            case RoPEAlgorithm::kYarn: {
                AM_ASSIGN_OR_RETURN(const double factor, ParseDouble(fields, "factor"));
                AM_ASSIGN_OR_RETURN(const int64_t original,
                                    ParseInt64(fields, "original_context_length"));
                AM_ASSIGN_OR_RETURN(const double beta_fast, ParseDouble(fields, "beta_fast"));
                AM_ASSIGN_OR_RETURN(const double beta_slow, ParseDouble(fields, "beta_slow"));
                AM_ASSIGN_OR_RETURN(const double rotary_output_scale,
                                    ParseDouble(fields, "rotary_output_scale"));
                AM_ASSIGN_OR_RETURN(const bool truncate,
                                    ParseBool(fields, "truncate_correction_range"));
                algorithm_params = YarnRoPE{.factor = factor,
                                            .original_context_length = original,
                                            .beta_fast = beta_fast,
                                            .beta_slow = beta_slow,
                                            .rotary_output_scale = rotary_output_scale,
                                            .truncate_correction_range = truncate};
                break;
            }
            case RoPEAlgorithm::kLlama3: {
                AM_ASSIGN_OR_RETURN(const double factor, ParseDouble(fields, "factor"));
                AM_ASSIGN_OR_RETURN(const double low,
                                    ParseDouble(fields, "low_frequency_factor"));
                AM_ASSIGN_OR_RETURN(const double high,
                                    ParseDouble(fields, "high_frequency_factor"));
                AM_ASSIGN_OR_RETURN(const int64_t original,
                                    ParseInt64(fields, "original_context_length"));
                algorithm_params = Llama3RoPE{.factor = factor,
                                              .low_frequency_factor = low,
                                              .high_frequency_factor = high,
                                              .original_context_length = original};
                break;
            }
            case RoPEAlgorithm::kLongRope: {
                AM_ASSIGN_OR_RETURN(const int64_t original,
                                    ParseInt64(fields, "original_context_length"));
                AM_ASSIGN_OR_RETURN(const double rotary_output_scale,
                                    ParseDouble(fields, "rotary_output_scale"));
                AM_ASSIGN_OR_RETURN(auto short_factors,
                                    ParseDoubleList(fields, "short_factors"));
                AM_ASSIGN_OR_RETURN(auto long_factors,
                                    ParseDoubleList(fields, "long_factors"));
                algorithm_params = LongRoPE{.short_factors = std::move(short_factors),
                                            .long_factors = std::move(long_factors),
                                            .original_context_length = original,
                                            .rotary_output_scale = rotary_output_scale};
                break;
            }
        }
        return OpParams{RoPEParams{.head_dim = *head_dim,
                                   .rotary_dim = *rotary_dim,
                                   .num_q_heads = *num_q_heads,
                                   .num_kv_heads = *num_kv_heads,
                                   .max_pos_embeddings = *max_pos_embeddings,
                                   .theta = *theta,
                                   .pairing = *pairing,
                                   .algorithm = std::move(algorithm_params)}};
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
        StatusOr<int64_t> num_q_heads = ParseInt64(fields, "num_q_heads");
        AM_RETURN_IF_ERROR(num_q_heads.status());
        StatusOr<int64_t> num_kv_heads = ParseInt64(fields, "num_kv_heads");
        AM_RETURN_IF_ERROR(num_kv_heads.status());
        StatusOr<int64_t> head_dim = ParseInt64(fields, "head_dim");
        AM_RETURN_IF_ERROR(head_dim.status());
        return OpParams{AttentionParams{.num_q_heads = *num_q_heads,
                                        .num_kv_heads = *num_kv_heads,
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

    if (kind == "Permute") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        const auto it = fields.find("permutation");
        if (it == fields.end()) {
            return Status::InvalidArgument(
                    "ParseOpParams: missing permutation field");
        }
        StatusOr<std::vector<uint32_t>> permutation = ParsePermutation(it->second);
        AM_RETURN_IF_ERROR(permutation.status());
        return OpParams{PermuteParams{.permutation = std::move(*permutation)}};
    }

    if (kind == "Reorder") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 0));
        return OpParams{ReorderParams{}};
    }

    if (kind == "QkvLinear") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 4));
        StatusOr<int64_t> q_out_features = ParseInt64(fields, "q_out_features");
        AM_RETURN_IF_ERROR(q_out_features.status());
        StatusOr<int64_t> k_out_features = ParseInt64(fields, "k_out_features");
        AM_RETURN_IF_ERROR(k_out_features.status());
        StatusOr<int64_t> v_out_features = ParseInt64(fields, "v_out_features");
        AM_RETURN_IF_ERROR(v_out_features.status());
        StatusOr<bool> has_bias = ParseBool(fields, "has_bias");
        AM_RETURN_IF_ERROR(has_bias.status());
        return OpParams{QkvLinearParams{.q_out_features = *q_out_features,
                                        .k_out_features = *k_out_features,
                                        .v_out_features = *v_out_features,
                                        .has_bias = *has_bias}};
    }

    if (kind == "GateUpLinear") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 3));
        StatusOr<int64_t> gate_out_features = ParseInt64(fields, "gate_out_features");
        AM_RETURN_IF_ERROR(gate_out_features.status());
        StatusOr<int64_t> up_out_features = ParseInt64(fields, "up_out_features");
        AM_RETURN_IF_ERROR(up_out_features.status());
        StatusOr<bool> has_bias = ParseBool(fields, "has_bias");
        AM_RETURN_IF_ERROR(has_bias.status());
        return OpParams{GateUpLinearParams{.gate_out_features = *gate_out_features,
                                           .up_out_features = *up_out_features,
                                           .has_bias = *has_bias}};
    }

    if (kind == "AddRmsNorm") {
        AM_RETURN_IF_ERROR(EnsureNoExtraFields(fields, 1));
        StatusOr<float> eps = ParseFloat(fields, "eps");
        AM_RETURN_IF_ERROR(eps.status());
        return OpParams{AddRmsNormParams{.eps = *eps}};
    }

    return Status::InvalidArgument("ParseOpParams: unknown parameter kind");
}

} // namespace aethermind
