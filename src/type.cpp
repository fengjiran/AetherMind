//
// Created by 赵丹 on 2025/8/13.
//

#include "type.h"

namespace aethermind {

const char* TypeKindToString(TypeKind kind) {
#define CASE(T)       \
    case TypeKind::T: \
        return #T;

    switch (kind) {
        FORALL_TYPES(CASE)
    }
#undef CASE
    return "";
}


}// namespace aethermind
