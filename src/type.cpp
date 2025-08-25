//
// Created by 赵丹 on 2025/8/13.
//

#include "type.h"
#include "type_traits.h"

namespace aethermind {

const char* TypeKindToString(TypeKind kind) {
#define CASE(T)       \
    case TypeKind::T: \
        return #T;

    switch (kind) {
        AETHERMIND_FORALL_TYPES(CASE)
    }
#undef CASE
    return "";
}

}// namespace aethermind
