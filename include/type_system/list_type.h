//
// Created by richard on 10/20/25.
//

#ifndef AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H

#include <utility>

#include "type_system/type.h"

namespace aethermind {

class ListType;
using ListTypePtr = std::shared_ptr<ListType>;
class ListType : public SingleElementType<ListType, TypeKind::ListType> {
public:
private:
    explicit ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H
