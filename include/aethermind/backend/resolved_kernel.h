#ifndef AETHERMIND_BACKEND_RESOLVED_KERNEL_H
#define AETHERMIND_BACKEND_RESOLVED_KERNEL_H

#include "aethermind/backend/kernel_types.h"
#include "aethermind/operators/op_type.h"

#include <cstddef>
#include <span>

namespace aethermind {

struct ResolvedKernel {
    OpType op_type = OpType::kUnknown;
    KernelFunc fn = nullptr;

    // attrs is plan-owned immutable metadata. Its lifetime must cover the whole
    // execution plan that references this resolved kernel, so plan builders must
    // either copy attrs into plan-owned storage or guarantee equivalent
    // immutability and lifetime externally.
    std::span<const std::byte> attrs{};

    // debug_name should refer to backend-owned stable storage (for example a
    // registered kernel descriptor string literal) and must stay valid for the
    // lifetime of the frozen execution plan.
    const char* debug_name = nullptr;
};

}// namespace aethermind

#endif
