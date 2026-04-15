#ifndef AETHERMIND_BACKEND_RESOLVED_KERNEL_H
#define AETHERMIND_BACKEND_RESOLVED_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>

namespace aethermind {

struct ResolvedKernel {
    OpType op_type = OpType::kUnknown;
    KernelFunc fn = nullptr;

    // attrs is plan-owned immutable metadata. Its lifetime must outlive the
    // frozen execution plan that references this resolved kernel.
    const void* attrs = nullptr;
    size_t attrs_size = 0;
    const char* debug_name = nullptr;
};

}// namespace aethermind

#endif
