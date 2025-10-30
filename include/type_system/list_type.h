//
// Created by richard on 10/20/25.
//

#ifndef AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H

#include "type_system/type.h"

namespace aethermind {

class ListType;
using ListTypePtr = std::shared_ptr<ListType>;
class ListType : public SingleElementType<ListType, TypeKind::ListType> {
public:
    String str() const override;

    // It's not exactly a singleton, but there should be exactly one instance of
    // List[T] for every T
    template<typename... T>
    static ListTypePtr Create(T&&... all) {
        return ListTypePtr(new ListType(std::forward<T>(all)...));//NOLINT
    }

    TypePtr CreateWithContainedTypes(const std::vector<TypePtr>&) const override;

private:
    explicit ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}

    String AnnotationImpl(const TypePrinter& printer) const override;
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H
