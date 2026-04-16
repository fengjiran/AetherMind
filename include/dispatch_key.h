//
// Created by 赵丹 on 2025/8/11.
//

#ifndef AETHERMIND_DISPATCH_KEY_H
#define AETHERMIND_DISPATCH_KEY_H

// ============================================================================
// FROZEN / DEPRECATED NOTICE
// ============================================================================
// This file is part of the legacy dispatch system and is NO LONGER the main
// dispatch path for new operator implementations.
//
// New dispatch mainline (since Batch 1-3 of dispatch redesign):
//   - OpType: operator semantic identity (include/aethermind/operators/op_type.h)
//   - KernelSelector: capability-based matching (include/aethermind/backend/kernel_selector.h)
//   - Backend-owned KernelRegistry with selector-based resolve
//
// This file is frozen and will NOT be extended. It is retained only for
// migration compatibility during the transition from OperatorName-based
// dispatch to OpType-centered dispatch.
//
// See: docs/designs/dispatch_design.md for the new mainline architecture.
// ============================================================================

#include <cstdint>
#include <ostream>
#include <string>

namespace aethermind {

#define FORALL_BACKEND_COMPONENTS(_, extra) \
    _(CPU, extra)                           \
    _(CUDA, extra)                          \
    _(CANN, extra)


#define FORALL_FUNCTIONALITY_KEYS(_) \
    _(Dense, )                       \
    _(Quantized, Quantized)          \
    _(Sparse, Sparse)                \
    _(SparseCsr, SparseCsr)          \
    _(NestedTensor, NestedTensor)    \
    _(AutogradFunctionality, Autograd)

enum class BackendComponent : uint8_t {
    InvalidBit = 0,

#define DEFINE_BACKEND_COMPONENT(n, _) n##Bit,
    FORALL_BACKEND_COMPONENTS(DEFINE_BACKEND_COMPONENT, unused)
#undef DEFINE_BACKEND_COMPONENT

            EndOfBackendKeys = CUDABit
};

enum class DispatchKey : uint16_t {
    Undefined = 0,
    CatchAll = Undefined,
    Dense,
    Quantized,
    Sparse,
    SparseCsr,
    NestedTensor,
    EndOfFunctionalityKeys,

};

static_assert(
        static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) +
                        static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys) <=
                64,
        "The BackendComponent and DispatchKey enums (below EndOfFunctionalityKeys)"
        " both map to backend and functionality bits"
        " into a 64-bit bitmask; you must have less than 64 total entries between them");


}// namespace aethermind

#endif// AETHERMIND_DISPATCH_KEY_H
