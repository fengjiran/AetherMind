#ifndef AETHERMIND_MODEL_MODEL_INSTANCE_BUILDER_H
#define AETHERMIND_MODEL_MODEL_INSTANCE_BUILDER_H

#include "aethermind/base/status.h"
#include "aethermind/model/formats/hf/hf_model_config.h"
#include "aethermind/model/model_weight_index.h"

#include <memory>

namespace aethermind {

class ModelInstance;

class ModelInstanceBuilder {
public:
    // Constructs a raw-only ModelInstance that holds config metadata and
    // resolved raw weight views.  The backing storage referenced by the
    // RawWeightView::storage shared pointers remains alive as long as the
    // returned ModelInstance exists.
    static StatusOr<std::unique_ptr<ModelInstance>> Create(
            HfModelConfig config,
            ModelWeightIndex raw_weight_index);
};

}// namespace aethermind

#endif
