//
// Created by richard on 11/23/25.
//

#ifndef AETHERMIND_TYPE_SYSTEM_TUPLE_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_TUPLE_TYPE_H

#include "type_system/type.h"


namespace aethermind {

class TupleType;
using TupleTypePtr = std::shared_ptr<TupleType>;
using NameList = std::vector<String>;
class TupleType : public NamedType {
public:
private:
    std::vector<TypePtr> elements_;
    bool has_free_variables_;

};

}

#endif//AETHERMIND_TYPE_SYSTEM_TUPLE_TYPE_H
