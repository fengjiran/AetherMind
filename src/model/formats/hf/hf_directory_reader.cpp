#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "aethermind/model/formats/hf/hf_safetensors_file.h"
#include "aethermind/model/formats/hf/hf_safetensors_index.h"
#include "aethermind/model/formats/hf/hf_utils.h"

#include <string>
#include <string_view>
#include <system_error>
#include <vector>

// Implements the HF directory front-end for Phase 1 model loading.
// The reader accepts only local safetensors layouts and keeps path validation
// close to weight loading so sharded indexes cannot redirect tensor reads
// outside the model directory.
namespace aethermind {
namespace {

// Parses the small subset of Hugging Face `config.json` required by the runtime.
// Unknown fields are intentionally skipped because upstream model configs often
// contain tokenizer, training, or architecture-specific metadata that Phase 1
// does not consume.
class ConfigJsonParser final : public hf::HfJsonReader {
public:
    explicit ConfigJsonParser(std::string_view input) noexcept
        : HfJsonReader(input) {}

    StatusOr<HfModelConfig> Parse() {
        if (!TryConsume('{')) {
            return Status::InvalidArgument("HF config must be a JSON object");
        }

        HfModelConfig config{};
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after HF config key"));
                AM_RETURN_IF_ERROR(ParseField(*key, config));
                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between HF config fields"));
            }
        }

        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument("HF config contains trailing JSON content");
        }

        ApplyDefaults(config);
        return config;
    }

private:
    Status ParseField(const std::string& key, HfModelConfig& config) {
        // Keep field-level diagnostics tied to the HF key that failed instead
        // of exposing only the low-level JSON parser error.
        const auto parse_into = [&](auto parse, auto& out) -> Status {
            auto value = parse();
            if (!value.ok()) {
                return FieldParseError(key, value.status());
            }
            out = std::move(*value);
            return Status::Ok();
        };

        if (key == "model_type") {
            return parse_into([&] { return ParseString(); }, config.model_type);
        }

        if (key == "architectures") {
            return parse_into([&] { return ParseStringArray(); }, config.architectures);
        }

        if (key == "hidden_size") {
            return parse_into([&] { return ParseInt64(); }, config.hidden_size);
        }

        if (key == "intermediate_size") {
            return parse_into([&] { return ParseInt64(); }, config.intermediate_size);
        }

        if (key == "num_hidden_layers") {
            return parse_into([&] { return ParseInt64(); }, config.num_hidden_layers);
        }

        if (key == "num_attention_heads") {
            return parse_into([&] { return ParseInt64(); }, config.num_attention_heads);
        }

        if (key == "num_key_value_heads") {
            return parse_into([&] { return ParseInt64(); }, config.num_key_value_heads);
        }

        if (key == "vocab_size") {
            return parse_into([&] { return ParseInt64(); }, config.vocab_size);
        }

        if (key == "max_position_embeddings") {
            return parse_into([&] { return ParseInt64(); }, config.max_position_embeddings);
        }

        if (key == "head_dim") {
            return parse_into([&] { return ParseInt64(); }, config.head_dim);
        }

        if (key == "rms_norm_eps") {
            return parse_into([&] { return ParseDouble(); }, config.rms_norm_eps);
        }

        if (key == "hidden_act") {
            return parse_into([&] { return ParseString(); }, config.hidden_act);
        }

        if (key == "tie_word_embeddings") {
            return parse_into([&] { return ParseBool(); }, config.tie_word_embeddings);
        }

        if (key == "attention_bias") {
            return parse_into([&] { return ParseBool(); }, config.attention_bias);
        }

        if (key == "mlp_bias") {
            return parse_into([&] { return ParseBool(); }, config.mlp_bias);
        }

        if (key == "rope_theta") {
            return parse_into([&] { return ParseDouble(); }, config.rope.theta);
        }

        if (key == "rope_scaling" || key == "rope_parameters") {
            auto value = ParseRopeConfig();
            if (!value.ok()) {
                return FieldParseError(key, value.status());
            }
            config.rope.scaling_factor = value->scaling_factor;
            config.rope.scaling_type = std::move(value->scaling_type);
            config.rope.theta = value->theta;
            return Status::Ok();
        }

        if (key == "torch_dtype" || key == "dtype") {
            auto value = ParseString();
            if (!value.ok()) {
                return FieldParseError(key, value.status());
            }
            config.weight_dtype_hint_name = *value;
            config.weight_dtype_hint = ParseTorchDType(*value);
            return Status::Ok();
        }
        return SkipValue();
    }

    StatusOr<HfRopeConfig> ParseRopeConfig() {
        if (TryConsumeLiteral("null")) {
            return HfRopeConfig{};
        }

        if (!TryConsume('{')) {
            return Status::InvalidArgument("Expected RoPE config object");
        }

        HfRopeConfig rope_config{};
        SkipWhitespace();
        if (!TryConsume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }

                AM_RETURN_IF_ERROR(Expect(':', "after RoPE config key"));
                if (*key == "factor") {
                    auto factor = ParseDouble();
                    if (!factor.ok()) {
                        return factor.status();
                    }
                    rope_config.scaling_factor = *factor;
                } else if (*key == "type" || *key == "rope_type") {
                    auto type = ParseString();
                    if (!type.ok()) {
                        return type.status();
                    }
                    if (*type != "default") {
                        rope_config.scaling_type = std::move(*type);
                    }
                } else if (*key == "rope_theta") {
                    auto theta = ParseDouble();
                    if (!theta.ok()) {
                        return theta.status();
                    }
                    rope_config.theta = *theta;
                } else {
                    AM_RETURN_IF_ERROR(SkipValue());
                }

                SkipWhitespace();
                if (TryConsume('}')) {
                    break;
                }
                AM_RETURN_IF_ERROR(Expect(',', "between RoPE config fields"));
            }
        }
        return rope_config;
    }

    static DataType ParseTorchDType(std::string_view dtype) {
        const auto is = [&](std::string_view value) noexcept {
            return dtype.compare(value) == 0;
        };

        if (is("auto")) {
            return DataType{};
        }
        if (is("float32") || is("float")) {
            return DataType::Float32();
        }
        if (is("float16") || is("half")) {
            return DataType::Float(16);
        }
        if (is("bfloat16")) {
            return DataType::BFloat(16);
        }
        if (is("float64") || is("double")) {
            return DataType::Double();
        }
        if (is("int8")) {
            return DataType::Int(8);
        }
        if (is("int4")) {
            return DataType::Int(4);
        }
        return DataType{};
    }

    static Status FieldParseError(const std::string& key, const Status& parse_status) {
        return Status(parse_status.code(),
                      "Failed to parse HF config field '" + key +
                              "': " + parse_status.message());
    }

    static void ApplyDefaults(HfModelConfig& config) {
        if (config.num_key_value_heads <= 0) {
            config.num_key_value_heads = config.num_attention_heads;
        }
    }
};

StatusOr<RawWeightTable> LoadSingleFileRawWeightTable(const HfDirectoryDescriptor& dir_desc) {
    auto safetensors_file = HfSafetensorsFile::Open(dir_desc.safetensors_path);
    if (!safetensors_file.ok()) {
        return Status(safetensors_file.status().code(),
                      hf::FormatPathMessage(safetensors_file.status().message(), dir_desc.model_dir));
    }

    RawWeightTable raw_weights;
    raw_weights.reserve(safetensors_file->Entries().size());
    for (const auto& entry: safetensors_file->Entries()) {
        raw_weights.emplace(entry.name, entry.view);
    }
    return raw_weights;
}

Status EnsureShardFileExists(const std::filesystem::path& shard_path) {
    std::error_code error;
    if (!std::filesystem::exists(shard_path, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to stat safetensors shard file", shard_path));
        }
        return Status::NotFound(
                hf::FormatPathMessage("Safetensors shard file not found", shard_path));
    }

    if (!std::filesystem::is_regular_file(shard_path, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to inspect safetensors shard file type", shard_path));
        }
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors shard path is not a regular file", shard_path));
    }
    return Status::Ok();
}

bool IsPathWithinDirectory(const std::filesystem::path& child,
                           const std::filesystem::path& parent) {
    // Lexically checks that `parent` is a prefix of `child`.
    // Returns true when child == parent (a directory is trivially "within" itself).
    // Callers that intend sub-path-only checks should verify child != parent separately.
    auto child_it = child.begin();
    for (auto parent_it = parent.begin(); parent_it != parent.end(); ++parent_it, ++child_it) {
        if (child_it == child.end() || *child_it != *parent_it) {
            return false;
        }
    }
    return true;
}

Status ValidateShardPathIsContained(const std::filesystem::path& canonical_model_dir,
                                    const std::filesystem::path& shard_path) {
    AM_RETURN_IF_ERROR(hf::RejectExistingPathIfSymlink(shard_path, "HF safetensors shard file"));

    // Canonicalize after rejecting symlinks so an index entry cannot escape the
    // model directory through either `..` components or symlink traversal.
    std::error_code error;
    const auto canonical_shard_path = std::filesystem::canonical(shard_path, error);
    if (error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to canonicalize safetensors shard path", shard_path));
    }

    if (!IsPathWithinDirectory(canonical_shard_path, canonical_model_dir)) {
        return Status::InvalidArgument(
                hf::FormatPathMessage("Safetensors shard path escapes HF model directory", shard_path));
    }
    return Status::Ok();
}

StatusOr<RawWeightTable> LoadShardedRawWeightTable(const HfDirectoryDescriptor& dir_desc) {
    auto index = HfSafetensorsIndex::Load(dir_desc.safetensors_index_path);
    if (!index.ok()) {
        return index.status();
    }

    const auto& weight_map = index->WeightMap();

    std::error_code canonical_error;
    const auto canonical_model_dir = std::filesystem::canonical(dir_desc.model_dir, canonical_error);
    if (canonical_error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to canonicalize HF model directory", dir_desc.model_dir));
    }

    RawWeightTable raw_weights;
    raw_weights.reserve(weight_map.size());

    for (const auto& shard_filename: index->UniqueShardFilenames()) {
        const auto shard_path = dir_desc.model_dir / shard_filename;
        AM_RETURN_IF_ERROR(EnsureShardFileExists(shard_path));
        AM_RETURN_IF_ERROR(ValidateShardPathIsContained(canonical_model_dir, shard_path));

        auto shard_file = HfSafetensorsFile::Open(shard_path);
        if (!shard_file.ok()) {
            return Status(shard_file.status().code(),
                          hf::FormatPathMessage(shard_file.status().message(), shard_path));
        }

        for (const auto& entry: shard_file->Entries()) {
            const auto expected_shard = weight_map.find(entry.name);
            if (expected_shard == weight_map.end()) {
                return Status::InvalidArgument(
                        hf::FormatPathMessage("Safetensors shard contains tensor not listed in index: '" + entry.name + "'",
                                              shard_path));
            }

            if (expected_shard->second != shard_filename) {
                return Status(StatusCode::kFailedPrecondition,
                              hf::FormatPathMessage("Safetensors index assigns tensor '" + entry.name +
                                                            "' to shard '" + expected_shard->second +
                                                            "' but it was found in shard '" + shard_filename + "'",
                                                    shard_path));
            }

            if (!raw_weights.emplace(entry.name, entry.view).second) {
                return Status::AlreadyExists(
                        hf::FormatPathMessage("Duplicate safetensors tensor across shards: '" + entry.name + "'",
                                              shard_path));
            }
        }
    }

    if (raw_weights.size() != weight_map.size()) {
        std::vector<std::string> missing;
        missing.reserve(weight_map.size() - raw_weights.size());
        for (const auto& [tensor_name, shard_filename]: weight_map) {
            if (!raw_weights.contains(tensor_name)) {
                std::string entry = "'";
                entry += tensor_name;
                entry += "' (assigned to shard '";
                entry += shard_filename;
                entry += "')";
                missing.push_back(std::move(entry));
            }
        }
        std::string message = "Safetensors index contains " +
                              std::to_string(missing.size()) +
                              " tensor(s) missing from their assigned shards: ";
        for (size_t i = 0; i < missing.size(); ++i) {
            if (i > 0) {
                message += ", ";
            }
            message += missing[i];
        }
        return Status::NotFound(
                hf::FormatPathMessage(message, dir_desc.model_dir));
    }
    return raw_weights;
}

}// namespace

StatusOr<HfDirectoryReader> HfDirectoryReader::Open(const std::filesystem::path& model_dir) {
    auto dir_desc = InspectDirectory(model_dir);
    if (!dir_desc.ok()) {
        return dir_desc.status();
    }

    return HfDirectoryReader(std::move(*dir_desc));
}

StatusOr<HfModelConfig> HfDirectoryReader::ParseConfig() const {
    auto text = hf::ReadFileText(dir_desc_.config_path);
    if (!text.ok()) {
        return text.status();
    }

    auto config = ConfigJsonParser(*text).Parse();
    if (!config.ok()) {
        return Status(config.status().code(),
                      hf::FormatPathMessage(config.status().message(), dir_desc_.config_path));
    }
    return config;
}

StatusOr<RawWeightTable> HfDirectoryReader::LoadRawWeightTable() const {
    if (dir_desc_.IsSingleFile()) {
        return LoadSingleFileRawWeightTable(dir_desc_);
    }

    if (dir_desc_.IsSharded()) {
        return LoadShardedRawWeightTable(dir_desc_);
    }

    return Status(StatusCode::kFailedPrecondition,
                  hf::FormatPathMessage("Unknown HF safetensors directory layout", dir_desc_.model_dir));
}

StatusOr<HfDirectoryDescriptor> HfDirectoryReader::InspectDirectory(const std::filesystem::path& model_dir) {
    // Reject an empty path before filesystem probing so the caller gets a usage
    // error instead of a platform-dependent filesystem diagnostic.
    if (model_dir.empty()) {
        return Status::InvalidArgument("HF model directory path must not be empty");
    }

    std::error_code error;
    // First validate the root directory. All later paths are interpreted relative
    // to this directory, so missing or non-directory roots fail before checking
    // any HF-specific files.
    if (!std::filesystem::exists(model_dir, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to stat HF model directory", model_dir));
        }
        return Status::NotFound(
                hf::FormatPathMessage("HF model directory not found", model_dir));
    }

    if (!std::filesystem::is_directory(model_dir, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to inspect HF model directory type", model_dir));
        }
        return Status::InvalidArgument(
                hf::FormatPathMessage("HF model path is not a directory", model_dir));
    }

    const auto config_path = model_dir / "config.json";
    const auto safetensors_path = model_dir / "model.safetensors";
    const auto safetensors_index_path = model_dir / "model.safetensors.index.json";

    // `config.json` is required for every supported HF layout because it provides
    // the architecture and tensor-shape metadata used before weight materialization.
    AM_RETURN_IF_ERROR(hf::RejectExistingPathIfSymlink(config_path, "HF config file"));
    if (!std::filesystem::exists(config_path, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to stat config.json", config_path));
        }
        return Status::NotFound(
                hf::FormatPathMessage("HF model directory is missing config.json", config_path));
    }

    if (!std::filesystem::is_regular_file(config_path, error) || error) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to inspect config.json type", config_path));
        }
        return Status::InvalidArgument(
                hf::FormatPathMessage("config.json path is not a regular file", config_path));
    }

    AM_RETURN_IF_ERROR(hf::RejectExistingPathIfSymlink(safetensors_path, "HF safetensors file"));
    const bool has_single_file = std::filesystem::exists(safetensors_path, error);
    if (error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to stat model.safetensors", safetensors_path));
    }

    AM_RETURN_IF_ERROR(hf::RejectExistingPathIfSymlink(safetensors_index_path, "HF safetensors index file"));
    const bool has_sharded_index = std::filesystem::exists(safetensors_index_path, error);
    if (error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to stat model.safetensors.index.json", safetensors_index_path));
    }

    // Treat simultaneous single-file and sharded layouts as ambiguous instead of
    // choosing one implicitly; otherwise stale files could silently change which
    // weights are loaded.
    if (has_single_file && has_sharded_index) {
        return Status(StatusCode::kFailedPrecondition,
                      hf::FormatPathMessage("HF model directory has conflicting single-file and sharded safetensors layouts",
                                            model_dir));
    }

    if (has_single_file && (!std::filesystem::is_regular_file(safetensors_path, error) || error)) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to inspect model.safetensors type", safetensors_path));
        }
        return Status::InvalidArgument(
                hf::FormatPathMessage("model.safetensors path is not a regular file", safetensors_path));
    }

    // The sharded path is only an index descriptor at this layer. Individual
    // shard files are validated later against the index before loading.
    if (has_sharded_index && (!std::filesystem::is_regular_file(safetensors_index_path, error) || error)) {
        if (error) {
            return Status::Internal(
                    hf::FormatPathMessage("Failed to inspect model.safetensors.index.json type", safetensors_index_path));
        }
        return Status::InvalidArgument(
                hf::FormatPathMessage("model.safetensors.index.json path is not a regular file", safetensors_index_path));
    }

    if (has_single_file) {
        return HfDirectoryDescriptor{
                .layout = HfDirectoryLayout::kSingleSafetensors,
                .model_dir = model_dir,
                .config_path = config_path,
                .safetensors_path = safetensors_path,
                .safetensors_index_path = {},
        };
    }

    if (has_sharded_index) {
        return HfDirectoryDescriptor{
                .layout = HfDirectoryLayout::kShardedSafetensors,
                .model_dir = model_dir,
                .config_path = config_path,
                .safetensors_path = {},
                .safetensors_index_path = safetensors_index_path,
        };
    }

    // A directory without either supported weight entry point is not a loadable
    // Phase 1 HF safetensors model, even if other HF files are present.
    return Status::NotFound(
            hf::FormatPathMessage("HF model directory is missing both model.safetensors and model.safetensors.index.json",
                                  model_dir));
}

}// namespace aethermind
