//
// Created by richard on 10/20/25.
//

#include "type_system/list_type.h"
#include "type_system/tensor_type.h"
#include "type_system/union_type.h"

#include <unordered_map>

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

TypePtr ListType::CreateWithContainedTypes(const std::vector<TypePtr>& contained_types) const {
    return Create(contained_types.at(0));
}

bool ListType::IsSubtypeOfImpl(const Type& other) const {
    if (Type::IsSubtypeOfImpl(other)) {
        return true;
    }

    if (other.kind() == AnyListType::Kind) {
        return true;
    }
    return false;
}

TypePtr ListType::Get(const String& identifier, const TypePtr& inner) {
    static std::unordered_map<std::tuple<String, TypePtr>, TypePtr> container_types;

    static std::mutex mutex;
    auto key = std::make_tuple(identifier, inner);
    std::lock_guard lock(mutex);
    if (!container_types.contains(key)) {
        TypePtr t = Create(inner);
        container_types.emplace(key, t);
    }
    return container_types[key];
}

ListTypePtr ListType::OfNumbers() {
    static ListTypePtr list_type = Create(NumberType::Global());
    return list_type;
}

ListTypePtr ListType::OfInts() {
    static ListTypePtr list_type = Create(IntType::Global());
    return list_type;
}

ListTypePtr ListType::OfFloats() {
    static ListTypePtr list_type = Create(FloatType::Global());
    return list_type;
}

ListTypePtr ListType::OfComplexDoubles() {
    static ListTypePtr list_type = Create(ComplexType::Global());
    return list_type;
}

ListTypePtr ListType::OfBools() {
    static ListTypePtr list_type = Create(BoolType::Global());
    return list_type;
}

ListTypePtr ListType::OfStrings() {
    static ListTypePtr list_type = Create(StringType::Global());
    return list_type;
}

ListTypePtr ListType::OfTensors() {
    static ListTypePtr list_type = Create(TensorType::Get());
    return list_type;
}

ListTypePtr ListType::OfOptionalTensors() {
    static ListTypePtr list_type = Create(OptionalType::OfTensor());
    return list_type;
}


}// namespace aethermind
