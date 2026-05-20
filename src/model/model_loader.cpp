#include "aethermind/model/model_loader.h"
#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/formats/hf/hf_model_validator.h"
#include "aethermind/model/formats/hf/hf_weight_resolver.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/model/model_instance_builder.h"
#include "aethermind/model/weight_prepack_planner.h"
#include "macros.h"

namespace aethermind {

StatusOr<std::unique_ptr<ModelInstance>> ModelLoader::Load(const ModelLoadOptions& options,
                                                           const Backend& backend,
                                                           const KernelRegistry& registry) {
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

    AM_RETURN_IF_ERROR(HfModelValidator::ValidateResolvedModel(*config, *resolved_weights));

    auto model = ModelInstanceBuilder::Create(std::move(*config), std::move(*resolved_weights));
    if (!model.ok()) {
        return model.status();
    }

    auto requests = WeightPrepackPlanner::BuildRequests(
            (*model)->GetConfig(), (*model)->GetResolvedWeights(), backend, registry);
    if (!requests.ok()) {
        return requests.status();
    }

    AM_RETURN_IF_ERROR(WeightPrepackPlanner::PrepackAndStore(**model, *requests));

    return model;
}

}// namespace aethermind
