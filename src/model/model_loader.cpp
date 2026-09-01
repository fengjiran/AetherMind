#include "aethermind/model/model_loader.h"
#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/formats/hf/hf_weight_resolver.h"
#include "aethermind/model/loaded_model.h"

#include <memory>
#include <utility>

namespace aethermind {

StatusOr<std::unique_ptr<LoadedModel>> ModelLoader::Load(
        const std::filesystem::path& model_dir) {
    auto reader = HfDirectoryReader::Open(model_dir);
    if (!reader.ok()) {
        return reader.status();
    }

    auto config = reader->ParseConfig();
    if (!config.ok()) {
        return config.status();
    }

    AM_RETURN_IF_ERROR(HfModelValidator::ValidateConfig(*config));

    auto raw_weights = reader->LoadRawWeightTable();
    if (!raw_weights.ok()) {
        return raw_weights.status();
    }

    AM_RETURN_IF_ERROR(HfModelValidator::ValidateWeightSet(*config, *raw_weights));

    auto resolved_weights = hf::ResolveWeights(*config, *raw_weights);
    if (!resolved_weights.ok()) {
        return resolved_weights.status();
    }

    AM_RETURN_IF_ERROR(HfModelValidator::ValidateResolvedModel(*config, *resolved_weights));

    return std::make_unique<LoadedModel>(std::move(*config), std::move(*resolved_weights));
}

} // namespace aethermind
