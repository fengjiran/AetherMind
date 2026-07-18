#ifndef AETHERMIND_BACKEND_RESOLVED_KERNEL_H
#define AETHERMIND_BACKEND_RESOLVED_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <vector>

namespace aethermind {

struct ResolvedKernel {
    OpType op_type = OpType::kUnknown;
    KernelFunc fn = nullptr;

    // attrs is immutable kernel metadata owned by the resolved kernel. Runtime
    // KernelContext instances borrow a span from this storage for each call.
    std::vector<std::byte> attrs{};

    // debug_name borrows backend-owned stable storage and must stay valid for
    // the lifetime of the frozen execution plan.
    const char* debug_name = nullptr;

    KernelParamsBuilder params_builder = nullptr;
    size_t params_size = 0;
};

}// namespace aethermind

#endif
