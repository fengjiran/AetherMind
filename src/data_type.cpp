//
// Created by 赵丹 on 25-7-2.
//

#include "data_type.h"

namespace aethermind {

#define DEFINE_MAKE(code, bits, lanes, name, T) \
    template<>                                  \
    DataType DataType::Make<T>() {              \
        return DataType(code, bits, lanes);     \
    }
SCALAR_TYPE_TO_NAME_AND_CPP_TYPE(DEFINE_MAKE);
#undef DEFINE_MAKE

}// namespace aethermind