//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "device.h"
#include "tensor.h"
#include "type_traits.h"

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

union Payload {
    union data {
        int64_t v_int;
        double v_double;
        bool v_bool;
        const char* v_str;
        void* v_handle;
        Device v_device;

        data() : v_int(0) {}
    } u;

    Tensor v_tensor;

    static_assert(std::is_trivially_copyable_v<data>);
    Payload() : u() {}
    Payload(const Payload&) = delete;
    Payload(Payload&&) = delete;
    Payload& operator=(const Payload&) = delete;
    Payload& operator=(Payload&&) = delete;

    ~Payload() {}
};


class Any {
public:
    Any() = default;

private:
    Payload payload_;
    Tag tag_;

#define COUNT_TAG(x) 1 +
    static constexpr auto kNumTags = AETHERMIND_FORALL_TAGS(COUNT_TAG) 0;
#undef COUNT_TAG
};

}// namespace aethermind

#endif//AETHERMIND_ANY_H
