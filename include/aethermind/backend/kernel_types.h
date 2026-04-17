#ifndef AETHERMIND_BACKEND_KERNEL_TYPES_H
#define AETHERMIND_BACKEND_KERNEL_TYPES_H

#include "aethermind/base/status.h"

namespace aethermind {

struct KernelInvocation;
struct OpKernelContext;
struct WorkspaceBinding;

using KernelFunc = Status (*)(const KernelInvocation&,
                              const OpKernelContext&,
                              const WorkspaceBinding&) noexcept;

}// namespace aethermind

#endif
