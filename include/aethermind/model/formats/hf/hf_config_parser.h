#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_CONFIG_PARSER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_CONFIG_PARSER_H

#include "aethermind/base/status.h"

#include <filesystem>
#include <string>

namespace aethermind {

struct HfRawConfig {
    std::string architecture{};
    std::string model_type{};
};

class HfConfigParser {
public:
    static StatusOr<HfRawConfig> ParseMinimal(
            const std::filesystem::path& config_path);
};

}// namespace aethermind

#endif
