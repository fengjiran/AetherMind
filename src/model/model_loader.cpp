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

    const auto layout = HfDirectoryReader::DiscoverLayout(options.model_dir);
    if (!layout.ok()) {
        return layout.status();
    }

    return Status(StatusCode::kUnimplemented,
                  "ModelLoader::Load is not implemented yet after layout discovery");
}

}// namespace aethermind
