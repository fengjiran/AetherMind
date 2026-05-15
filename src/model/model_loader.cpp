#include "aethermind/model/model_loader.h"
#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/formats/hf/hf_weight_resolver.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/model/model_instance_builder.h"
#include "macros.h"

namespace aethermind {

StatusOr<std::unique_ptr<ModelInstance>> ModelLoader::Load(
        const ModelLoadOptions& options,
        const Backend& backend,
        const KernelRegistry& registry) {
    UNUSED(backend);
    UNUSED(registry);

    auto reader = HfDirectoryReader::Open(options.model_dir);
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
    return ModelInstanceBuilder::Create(std::move(*config), std::move(*resolved_weights));
}

}// namespace aethermind
