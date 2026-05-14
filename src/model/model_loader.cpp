#include "aethermind/model/model_loader.h"
#include "aethermind/model/formats/hf/hf_directory_reader.h"
#include "aethermind/model/model_instance.h"
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

    auto tensor_table = reader->LoadTensorTable();
    if (!tensor_table.ok()) {
        return tensor_table.status();
    }

    return Status(StatusCode::kUnimplemented,
                  "ModelLoader::Load reached reader tensor table; ModelInstance build is not implemented yet");
}

}// namespace aethermind
