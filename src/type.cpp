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
        FORALL_TYPES(CASE)
    }
#undef CASE
    return "";
}

std::string TagToString(Tag t) {
#define CASE(T)  \
    case Tag::T: \
        return #T;

    switch (t) {
        AETHERMIND_FORALL_TAGS(CASE)
    }
#undef CASE
    return "";
}


}// namespace aethermind
