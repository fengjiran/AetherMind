//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "device.h"
#include "tensor.h"
#include "type_traits.h"

namespace aethermind {

class Any {
public:
    Any() = default;

private:
    AetherMindAny data_;

#define COUNT_TAG(x) 1 +
    static constexpr auto kNumTags = AETHERMIND_FORALL_TAGS(COUNT_TAG) 0;
#undef COUNT_TAG
};

}// namespace aethermind

#endif//AETHERMIND_ANY_H
