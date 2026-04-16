//
// Created by 赵丹 on 2025/8/11.
//

#ifndef AETHERMIND_DISPATCH_KEY_SET_H
#define AETHERMIND_DISPATCH_KEY_SET_H

// ============================================================================
// FROZEN / DEPRECATED NOTICE
// ============================================================================
// This file is part of the legacy dispatch system and is NO LONGER the main
// dispatch path for new operator implementations.
//
// New dispatch mainline uses KernelSelector for capability-based matching
// instead of bitset-based DispatchKeySet. The selector-based approach:
//   - Matches on DeviceType, DataType, WeightFormat, IsaLevel, ExecPhase
//   - Supports priority-based selection among multiple implementations
//   - Backend-owned, no global registry
//
// This file is frozen and will NOT be extended. It is retained only for
// migration compatibility.
//
// See: docs/designs/dispatch_design.md for the new mainline architecture.
// ============================================================================

#include "dispatch_key.h"

namespace aethermind {

class DispatchKeySet final {
public:
    DispatchKeySet() = default;


private:
    DispatchKeySet(uint64_t repr) : repr_(repr) {}
    uint64_t repr_ = 0;
};

}// namespace aethermind

#endif// AETHERMIND_DISPATCH_KEY_SET_H
