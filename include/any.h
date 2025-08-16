//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "device.h"
#include "tensor.h"

namespace aethermind {

#define AETHERMIND_FORALL_TAGS(_) \
    _(None)                       \
    _(Tensor)                     \
    _(Storage)                    \
    _(Double)                     \
    _(ComplexDouble)              \
    _(Int)                        \
    _(SymInt)                     \
    _(SymFloat)                   \
    _(SymBool)                    \
    _(Bool)                       \
    _(Tuple)                      \
    _(String)                     \
    _(Blob)                       \
    _(GenericList)                \
    _(GenericDict)                \
    _(Future)                     \
    _(Await)                      \
    _(Device)                     \
    _(Stream)                     \
    _(Object)                     \
    _(PyObject)                   \
    _(Uninitialized)              \
    _(Capsule)                    \
    _(RRef)                       \
    _(Quantizer)                  \
    _(Generator)                  \
    _(Enum)

enum class Tag : uint32_t {
#define DEFINE_TAG(x) x,
    AETHERMIND_FORALL_TAGS(DEFINE_TAG)
#undef DEFINE_TAG
};



class Any {
public:


private:
    union data {
        int64_t v_int{};
        double v_double;
        bool v_bool;
        Device v_device;
        Tensor v_tensor;
        void* v_handle;

        data() : v_int(0) {}
    } data_;

    Tag tag_;
};

}// namespace aethermind

#endif//AETHERMIND_ANY_H
