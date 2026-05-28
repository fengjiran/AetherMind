#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"

namespace aethermind {

struct KernelContext;
struct KernelInvocation;
struct WorkspaceBinding;

using KernelFunc = Status (*)(const KernelInvocation&,
                              const KernelContext&,
                              const WorkspaceBinding&) noexcept;

}// namespace aethermind

#endif
