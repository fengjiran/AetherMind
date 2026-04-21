#include "aethermind/model/formats/hf/hf_config_parser.h"

#include "macros.h"

namespace aethermind {

StatusOr<HfRawConfig> HfConfigParser::ParseMinimal(
        const std::filesystem::path& config_path) {
    UNUSED(config_path);
    return Status(StatusCode::kUnimplemented,
                  "HfConfigParser::ParseMinimal is not implemented yet");
}

}// namespace aethermind
