#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"

namespace aethermind {

struct KernelKey {};

using KernelFn = Status (*)() noexcept;

}// namespace aethermind

#endif
