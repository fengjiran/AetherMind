#ifndef AETHERMIND_MODEL_FORMATS_HF_HF_CONFIG_PARSER_H
#define AETHERMIND_MODEL_FORMATS_HF_HF_CONFIG_PARSER_H

#include "aethermind/base/status.h"
#include "aethermind/model/model_config.h"

#include <filesystem>

namespace aethermind {

class HfConfigParser {
public:
    static StatusOr<ModelConfig> ParseConfigFile(const std::filesystem::path& config_path);
};

}// namespace aethermind

#endif
