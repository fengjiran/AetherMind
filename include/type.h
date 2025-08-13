//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_H
#define AETHERMIND_TYPE_H

#include "macros.h"

#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace aethermind {

#define FORALL_TYPES(_) \
    _(AnyType)          \
    _(EnumType)         \
    _(AnyEnumType)      \
    _(TensorType)       \
    _(StorageType)      \
    _(TupleType)        \
    _(ListType)         \
    _(DictType)         \
    _(NumberType)       \
    _(FloatType)        \
    _(ComplexType)      \
    _(FutureType)       \
    _(AwaitType)        \
    _(RRefType)         \
    _(IntType)          \
    _(NoneType)         \
    _(StringType)       \
    _(GeneratorType)    \
    _(QuantizerType)    \
    _(BoolType)         \
    _(OptionalType)     \
    _(VarType)          \
    _(DeviceObjType)    \
    _(StreamObjType)    \
    _(FunctionType)     \
    _(ClassType)        \
    _(PyObjectType)     \
    _(CapsuleType)      \
    _(InterfaceType)    \
    _(QSchemeType)      \
    _(ScalarTypeType)   \
    _(LayoutType)       \
    _(MemoryFormatType) \
    _(AnyListType)      \
    _(AnyTupleType)     \
    _(AnyClassType)     \
    _(SymIntType)       \
    _(SymFloatType)     \
    _(SymBoolType)      \
    _(UnionType)        \
    _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
    FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

const char* TypeKindToString(TypeKind kind);

}// namespace aethermind

#endif//AETHERMIND_TYPE_H
