//
// Created by 赵丹 on 2025/8/11.
//

#ifndef AETHERMIND_DISPATCH_KEY_H
#define AETHERMIND_DISPATCH_KEY_H

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
    EndOfFunctionalityKeys
};

static_assert(
        static_cast<uint8_t>(BackendComponent::EndOfBackendKeys) +
                        static_cast<uint8_t>(DispatchKey::EndOfFunctionalityKeys) <=
                64,
        "The BackendComponent and DispatchKey enums (below EndOfFunctionalityKeys)"
        " both map to backend and functionality bits"
        " into a 64-bit bitmask; you must have less than 64 total entries between them");


}// namespace aethermind

#endif//AETHERMIND_DISPATCH_KEY_H
