#include "aethermind/model/model_loader.h"
#include "aethermind/model/formats/hf/hf_config_parser.h"
#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/model_instance.h"
#include "aethermind/model/model_validator.h"
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

    auto config = HfConfigParser::ParseConfigFile(reader->Layout().config_path);
    if (!config.ok()) {
        return config.status();
    }

    AM_RETURN_IF_ERROR(ModelValidator::ValidateConfig(*config));

    auto tensor_table = reader->LoadTensorTable();
    if (!tensor_table.ok()) {
        return tensor_table.status();
    }

    AM_RETURN_IF_ERROR(ModelValidator::ValidateTensorSet(*config, *tensor_table));

    return Status(StatusCode::kUnimplemented,
                  "ModelLoader::Load reached validated config and tensor table; ModelInstance build is not implemented yet");
}

}// namespace aethermind
