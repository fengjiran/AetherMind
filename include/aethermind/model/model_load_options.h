#ifndef AETHERMIND_MODEL_MODEL_LOAD_OPTIONS_H
#define AETHERMIND_MODEL_MODEL_LOAD_OPTIONS_H

#include <filesystem>

namespace aethermind {

struct ModelLoadOptions {
    std::filesystem::path model_dir{};
};

}// namespace aethermind

#endif
