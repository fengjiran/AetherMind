//
// Created by richard on 10/20/25.
//

#include "type_system/list_type.h"
#include "type_system/tensor_type.h"

namespace aethermind {

String ListType::str() const {
    std::stringstream ss;
    ss << GetElementType()->str() << "[]";
    return ss.str();
}

String ListType::AnnotationImpl(const TypePrinter& printer) const {
    std::stringstream ss;
    ss << "List[" << GetElementType()->Annotation(printer) << "]";
    return ss.str();
}


}// namespace aethermind
