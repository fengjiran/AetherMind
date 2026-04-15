//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_INVOCATION_H
#define AETHERMIND_BACKEND_KERNEL_INVOCATION_H

#include "aethermind/backend/kernel_selector.h"
#include "aethermind/operators/op_type.h"

namespace aethermind {

struct KernelInvocation {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
};

}// namespace aethermind
#endif
