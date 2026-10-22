#ifndef AETHERMIND_C_API_RUNTIME_INTERNAL_H
#define AETHERMIND_C_API_RUNTIME_INTERNAL_H

#include "aethermind/base/status.h"

namespace aethermind {

struct LoweredModelArtifact;
class Runtime;

namespace c_api::detail {

/// @brief Validates an existing Runtime against the complete compiled KV state.
/// @note Adapter-private and read-only; does not prepare a model, modify KV, or
///       replace the compiler's semantic/structural validation.
Status ValidateRuntimeModelKV(const Runtime& runtime, const LoweredModelArtifact& artifact);

} // namespace c_api::detail
} // namespace aethermind

#endif
