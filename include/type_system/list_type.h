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
    template<typename... Args>
    static ListTypePtr Create(Args&&... args) {
        return ListTypePtr(new ListType(std::forward<Args>(args)...));//NOLINT
    }

    TypePtr CreateWithContainedTypes(const std::vector<TypePtr>&) const override;

    bool IsSubtypeOfExtTypeImpl(const Type& other, std::ostream* why_not) const override;

    static TypePtr Get(const String& identifier, const TypePtr& inner);

    static ListTypePtr OfNumbers();

    static ListTypePtr OfInts();

    static ListTypePtr OfFloats();

    static ListTypePtr OfBools();

    static ListTypePtr OfComplexDoubles();

    static ListTypePtr OfStrings();

    static ListTypePtr OfTensors();

    static ListTypePtr OfOptionalTensors();

private:
    explicit ListType(TypePtr elem) : SingleElementType(std::move(elem)) {}

    String AnnotationImpl(const TypePrinter& printer) const override;
};

// the common supertype of all lists,
// List[T] <: AnyList for all T
class AnyListType;
using AnyListTypePtr = SingletonTypePtr<AnyListType>;
class AnyListType : public Singleton<AnyListType> {
public:
    NODISCARD bool Equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    NODISCARD String str() const override {
        return "list";
    }

    static constexpr auto Kind = TypeKind::AnyListType;

private:
    AnyListType() : Singleton(Kind) {}
    friend class Singleton;
};

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_LIST_TYPE_H
