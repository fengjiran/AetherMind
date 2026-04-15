#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"

namespace aethermind {

struct KernelKey;

// Phase 3 minimal placeholder.
// Will be widened in later phases to include invocation/context/workspace.
using KernelFunc = Status (*)() noexcept;

}// namespace aethermind

#endif
