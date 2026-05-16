#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"
#include "aethermind/model/formats/hf/hf_safetensors_file.h"
#include "aethermind/model/formats/hf/hf_utils.h"

#include <string>
#include <system_error>
#include <vector>

namespace aethermind {
namespace {

class ConfigJsonParser final : public hf::HfJsonReader {
public:
    explicit ConfigJsonParser(std::string_view input) noexcept
        : HfJsonReader(input) {}

    StatusOr<HfModelConfig> Parse() {
        if (!Consume('{')) {
            return Status::InvalidArgument("HF config must be a JSON object");
        }

        HfModelConfig config{};
        SkipWhitespace();
        if (!Consume('}')) {
            while (true) {
                const auto key = ParseString();
                if (!key.ok()) {
                    return key.status();
                }
                if (!Expect(':')) {
                    return Status::InvalidArgument("Expected ':' after HF config key");
                }
                AM_RETURN_IF_ERROR(ParseField(*key, config));
                SkipWhitespace();
                if (Consume('}')) {
                    break;
                }
                if (!Expect(',')) {
                    return Status::InvalidArgument("Expected ',' between HF config fields");
                }
            }
        }

        SkipWhitespace();
        if (!AtEnd()) {
            return Status::InvalidArgument("HF config contains trailing JSON content");
        }

        AM_RETURN_IF_ERROR(ValidateRequiredFields(config));
        ApplyDefaults(config);
        return config;
    }

private:
    Status ParseField(const std::string& key, HfModelConfig& config) {
        if (key == "model_type") {
            auto v = ParseString();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.model_type = std::move(*v);
            return Status::Ok();
        }

        if (key == "architectures") {
            auto v = ParseStringArray();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.architectures = std::move(*v);
            return Status::Ok();
        }

        if (key == "hidden_size") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.hidden_size = *v;
            return Status::Ok();
        }

        if (key == "intermediate_size") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.intermediate_size = *v;
            return Status::Ok();
        }

        if (key == "num_hidden_layers") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.num_hidden_layers = *v;
            return Status::Ok();
        }

        if (key == "num_attention_heads") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.num_attention_heads = *v;
            return Status::Ok();
        }

        if (key == "num_key_value_heads") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.num_key_value_heads = *v;
            return Status::Ok();
        }

        if (key == "vocab_size") {
            auto v = ParseInt64();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.vocab_size = *v;
            return Status::Ok();
        }

        if (key == "rms_norm_eps") {
            auto v = ParseDouble();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.rms_norm_eps = *v;
            return Status::Ok();
        }

        if (key == "tie_word_embeddings") {
            auto v = ParseBool();
            if (!v.ok()) return FieldParseError(key, v.status());
            config.tie_word_embeddings = *v;
            return Status::Ok();
        }
        return SkipValue();
    }

    static Status FieldParseError(const std::string& key, const Status& parse_status) {
        return Status(parse_status.code(),
                      "Failed to parse HF config field '" + key +
                              "': " + parse_status.message());
    }

    static Status ValidateRequiredFields(const HfModelConfig& config) {
        if (config.model_type.empty()) {
            return Status::InvalidArgument("HF config is missing required 'model_type' field");
        }
        if (config.architectures.empty()) {
            return Status::InvalidArgument("HF config is missing required 'architectures' field");
        }
        if (config.hidden_size <= 0) {
            return Status::InvalidArgument("HF config is missing required 'hidden_size' field");
        }
        if (config.intermediate_size <= 0) {
            return Status::InvalidArgument("HF config is missing required 'intermediate_size' field");
        }
        if (config.num_hidden_layers <= 0) {
            return Status::InvalidArgument("HF config is missing required 'num_hidden_layers' field");
        }
        if (config.num_attention_heads <= 0) {
            return Status::InvalidArgument("HF config is missing required 'num_attention_heads' field");
        }
        if (config.vocab_size <= 0) {
            return Status::InvalidArgument("HF config is missing required 'vocab_size' field");
        }
        if (config.rms_norm_eps <= 0.0) {
            return Status::InvalidArgument("HF config is missing required 'rms_norm_eps' field");
        }
        return Status::Ok();
    }

    static void ApplyDefaults(HfModelConfig& config) {
        if (config.num_key_value_heads <= 0) {
            config.num_key_value_heads = config.num_attention_heads;
        }
    }
};

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
    if (!dir_desc_.IsSingleFile()) {
        return Status(StatusCode::kUnimplemented,
                      hf::FormatPathMessage("Only single-file HF safetensors layout is implemented", dir_desc_.model_dir));
    }

    auto safetensors_file = HfSafetensorsFile::Open(dir_desc_.safetensors_path);
    if (!safetensors_file.ok()) {
        return Status(safetensors_file.status().code(),
                      hf::FormatPathMessage(safetensors_file.status().message(), dir_desc_.model_dir));
    }

    RawWeightTable raw_weights;
    raw_weights.reserve(safetensors_file->Entries().size());
    for (const auto& entry: safetensors_file->Entries()) {
        raw_weights.emplace(entry.name, entry.view);
    }
    return raw_weights;
}

StatusOr<HfDirectoryDescriptor> HfDirectoryReader::InspectDirectory(const std::filesystem::path& model_dir) {
    if (model_dir.empty()) {
        return Status::InvalidArgument("HF model directory path must not be empty");
    }

    std::error_code error;
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

    const bool has_single_file = std::filesystem::exists(safetensors_path, error);
    if (error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to stat model.safetensors", safetensors_path));
    }

    const bool has_sharded_index = std::filesystem::exists(safetensors_index_path, error);
    if (error) {
        return Status::Internal(
                hf::FormatPathMessage("Failed to stat model.safetensors.index.json", safetensors_index_path));
    }

    if (has_single_file && has_sharded_index) {
        return Status(StatusCode::kFailedPrecondition,
                      hf::FormatPathMessage("HF model directory has conflicting single-file and sharded safetensors layouts",
                                            model_dir));
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
        return Status(StatusCode::kUnimplemented,
                      hf::FormatPathMessage("Sharded HF safetensors layout is not implemented yet", model_dir));
    }

    return Status::NotFound(
            hf::FormatPathMessage("HF model directory is missing both model.safetensors and model.safetensors.index.json",
                                  model_dir));
}

}// namespace aethermind
