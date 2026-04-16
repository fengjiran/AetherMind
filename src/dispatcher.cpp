//
// Created by 赵丹 on 2025/8/13.
//

// ============================================================================
// FROZEN / DEPRECATED NOTICE
// ============================================================================
// This file implements the legacy global Dispatcher. It is NO LONGER the
// main dispatch path for new operator implementations.
//
// New operators must use:
//   - Backend::ResolveKernel(OpType, KernelSelector) for kernel lookup
//   - ExecutionPlanBuilder for plan-build-time resolve
//   - ResolvedKernel for frozen execution metadata
//
// Do NOT add new runtime resolve logic here. This implementation is frozen.
// ============================================================================

#include "dispatcher.h"

namespace aethermind {

std::vector<OperatorName> Dispatcher::ListAllOpNames() {
    std::vector<OperatorName> names;
    names.reserve(table_.size());
    for (const auto& [name, _]: table_) {
        names.emplace_back(name);
    }
    return names;
}


}// namespace aethermind
