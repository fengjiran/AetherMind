#include "aethermind/model/formats/hf/hf_config_parser.h"
#include "aethermind/model/formats/hf/hf_format_utils.h"
#include "aethermind/model/formats/hf/hf_json_reader.h"

#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace aethermind {
namespace {

StatusOr<std::string> ReadConfigText(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error) || error) {
        if (error) {
            return Status::Internal(hf::FormatPathMessage("Failed to stat config.json", path));
        }
        return Status::NotFound(hf::FormatPathMessage("config.json not found", path));
    }

    if (!std::filesystem::is_regular_file(path, error) || error) {
        if (error) {
            return Status::Internal(hf::FormatPathMessage("Failed to inspect config.json type", path));
        }
        return Status::InvalidArgument(hf::FormatPathMessage("config.json path is not a regular file", path));
    }

    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream.is_open()) {
        return Status::Internal(hf::FormatPathMessage("Failed to open config.json", path));
    }

    const auto end_pos = stream.tellg();
    if (end_pos < 0) {
        return Status::Internal(hf::FormatPathMessage("Failed to determine config.json size", path));
    }

    std::string text(end_pos, '\0');
    stream.seekg(0, std::ios::beg);
    if (!text.empty()) {
        stream.read(text.data(), static_cast<std::streamsize>(text.size()));
        if (!stream) {
            return Status::Internal(hf::FormatPathMessage("Failed to read config.json", path));
        }
    }
    return text;
}

class ConfigJsonParser final : public hf::HfJsonReader {
public:
    explicit ConfigJsonParser(std::string_view input) noexcept
        : HfJsonReader(input) {}

    StatusOr<ModelConfig> Parse() {
        if (!Consume('{')) {
            return Status::InvalidArgument("HF config must be a JSON object");
        }

        ModelConfig config;
        std::optional<std::string> model_type;
        std::optional<std::vector<std::string>> architectures;
        std::optional<int64_t> hidden_size;
        std::optional<int64_t> intermediate_size;
        std::optional<int64_t> num_hidden_layers;
        std::optional<int64_t> num_attention_heads;
        std::optional<int64_t> num_key_value_heads;
        std::optional<int64_t> vocab_size;
        std::optional<double> rms_norm_eps;
        std::optional<bool> tie_word_embeddings;

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

                if (*key == "model_type") {
                    auto value = ParseString();
                    if (!value.ok()) {
                        return value.status();
                    }
                    model_type = std::move(*value);
                } else if (*key == "architectures") {
                    auto value = ParseStringArray();
                    if (!value.ok()) {
                        return value.status();
                    }
                    architectures = std::move(*value);
                } else if (*key == "hidden_size") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    hidden_size = *value;
                } else if (*key == "intermediate_size") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    intermediate_size = *value;
                } else if (*key == "num_hidden_layers") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    num_hidden_layers = *value;
                } else if (*key == "num_attention_heads") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    num_attention_heads = *value;
                } else if (*key == "num_key_value_heads") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    num_key_value_heads = *value;
                } else if (*key == "vocab_size") {
                    auto value = ParseInt64();
                    if (!value.ok()) {
                        return value.status();
                    }
                    vocab_size = *value;
                } else if (*key == "rms_norm_eps") {
                    auto value = ParseDouble();
                    if (!value.ok()) {
                        return value.status();
                    }
                    rms_norm_eps = *value;
                } else if (*key == "tie_word_embeddings") {
                    auto value = ParseBool();
                    if (!value.ok()) {
                        return value.status();
                    }
                    tie_word_embeddings = *value;
                } else {
                    AM_RETURN_IF_ERROR(SkipValue());
                }

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

        if (!model_type.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'model_type' field");
        }

        if (!architectures.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'architectures' field");
        }

        if (!hidden_size.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'hidden_size' field");
        }

        if (!intermediate_size.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'intermediate_size' field");
        }

        if (!num_hidden_layers.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'num_hidden_layers' field");
        }

        if (!num_attention_heads.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'num_attention_heads' field");
        }

        if (!vocab_size.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'vocab_size' field");
        }

        if (!rms_norm_eps.has_value()) {
            return Status::InvalidArgument("HF config is missing required 'rms_norm_eps' field");
        }

        config.model_type = std::move(*model_type);
        config.architectures = std::move(*architectures);
        config.hidden_size = *hidden_size;
        config.intermediate_size = *intermediate_size;
        config.num_hidden_layers = *num_hidden_layers;
        config.num_attention_heads = *num_attention_heads;
        config.num_key_value_heads = num_key_value_heads.value_or(*num_attention_heads);
        config.vocab_size = *vocab_size;
        config.rms_norm_eps = *rms_norm_eps;
        config.tie_word_embeddings = tie_word_embeddings.value_or(false);
        return config;
    }
};

}// namespace

namespace hf {

StatusOr<ModelConfig> ParseConfigFile(const std::filesystem::path& config_path) {
    auto text = ReadConfigText(config_path);
    if (!text.ok()) {
        return text.status();
    }

    auto config = ConfigJsonParser(*text).Parse();
    if (!config.ok()) {
        return Status(config.status().code(),
                      hf::FormatPathMessage(config.status().message(), config_path));
    }
    return config;
}

}// namespace hf

}// namespace aethermind
