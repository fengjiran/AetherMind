//
// Created by richard on 10/1/25.
//

#ifndef AETHERMIND_FUNCTION_SCHEMA_H
#define AETHERMIND_FUNCTION_SCHEMA_H

#include "any.h"
#include "container/string.h"
#include "type.h"

#include <optional>

namespace aethermind {

class Argument {
public:
    //
private:
    String name_;
    TypePtr type_;
    std::optional<int32_t> N_;
    std::optional<Any> default_value_;
};

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_SCHEMA_H
