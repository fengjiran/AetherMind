//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_H
#define AETHERMIND_TYPE_H

#include "macros.h"
#include "type_ptr.h"
#include "container/array_view.h"

#include <functional>
#include <glog/logging.h>
#include <memory>
#include <optional>
#include <string>

namespace aethermind {

#define AETHERMIND_FORALL_TYPES(_) \
    _(AnyType)                     \
    _(EnumType)                    \
    _(AnyEnumType)                 \
    _(TensorType)                  \
    _(StorageType)                 \
    _(TupleType)                   \
    _(ListType)                    \
    _(DictType)                    \
    _(NumberType)                  \
    _(FloatType)                   \
    _(ComplexType)                 \
    _(FutureType)                  \
    _(AwaitType)                   \
    _(RRefType)                    \
    _(IntType)                     \
    _(NoneType)                    \
    _(StringType)                  \
    _(GeneratorType)               \
    _(QuantizerType)               \
    _(BoolType)                    \
    _(OptionalType)                \
    _(VarType)                     \
    _(DeviceObjType)               \
    _(StreamObjType)               \
    _(FunctionType)                \
    _(ClassType)                   \
    _(PyObjectType)                \
    _(CapsuleType)                 \
    _(InterfaceType)               \
    _(QSchemeType)                 \
    _(ScalarTypeType)              \
    _(LayoutType)                  \
    _(MemoryFormatType)            \
    _(AnyListType)                 \
    _(AnyTupleType)                \
    _(AnyClassType)                \
    _(SymIntType)                  \
    _(SymFloatType)                \
    _(SymBoolType)                 \
    _(UnionType)                   \
    _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
    AETHERMIND_FORALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

const char* TypeKindToString(TypeKind kind);

class Type;
class SharedType;

// Use this to customize how a Type is printed using `annotation_str()`. If
// std::nullopt is returned, `annotation_str()` falls through to its default
// implementation.
using TypePrinter = std::function<std::optional<std::string>(const Type&)>;

namespace detail {

template<typename T>
struct IsSingletonType : std::false_type {};

template<typename T>
constexpr bool is_singleton_type_v = IsSingletonType<T>::value;

template<typename T, typename Enable = void>
struct CastReturnType {
    using type = std::shared_ptr<T>;
};

template<typename T, typename Enable = void>
struct CastConstReturnType {
    using type = std::shared_ptr<const T>;
};

template<typename T>
struct CastReturnType<T, std::enable_if_t<is_singleton_type_v<T>>> {
    using type = SingletonTypePtr<T>;
};

template<typename T>
struct CastConstReturnType<T, std::enable_if_t<is_singleton_type_v<T>>> {
    using type = SingletonTypePtr<const T>;
};

template<typename T, typename Enable = void>
using CastReturnType_t = CastReturnType<T, Enable>::type;

template<typename T, typename Enable = void>
using CastConstReturnType_t = CastConstReturnType<T, Enable>::type;

template<typename T>
struct as_shared_type {
    using type = SharedType*;
};

template<typename T>
struct as_shared_type<const T*> {
    using type = const SharedType*;
};

}// namespace detail


#define DECLARE_SINGLETON_TYPE(Type)                  \
    class Type;                                       \
    namespace detail {                                \
    template<>                                        \
    struct IsSingletonType<Type> : std::true_type {}; \
    }

DECLARE_SINGLETON_TYPE(AnyType);
DECLARE_SINGLETON_TYPE(NumberType);
DECLARE_SINGLETON_TYPE(IntType);
DECLARE_SINGLETON_TYPE(FloatType);

using TypePtr = SingletonOrSharedTypePtr<Type>;

class Type {
public:
    TypeKind kind() const {
        return kind_;
    }

    virtual std::string str() const = 0;

    // a == b
    virtual bool equals(const Type& rhs) const = 0;

    // a == b <=> b == a
    virtual bool symmetric() const {
        return true;
    }

    virtual bool isUnionType() const {
        return false;
    }

    // list of types this type contains, e.g. for a List then element type of
    // list for a tuple, the types of the tuple elements
    virtual ArrayView<TypePtr> containedTypes() const {
        return ArrayView<TypePtr>();
    }

    virtual TypePtr containedType(size_t i) const {
        return containedTypes().at(i);
    }

    virtual size_t containedTypeSize() const {
        return containedTypes().size();
    }

    virtual bool hasFreeVars() const {
        return false;
    }

    std::string annotation_str(const TypePrinter& printer) const {
        if (printer) {
            if (auto renamed = printer(*this)) {
                return *renamed;
            }
        }
        return this->annotation_str_impl(printer);
    }

    std::string annotation_str() const {
        // Overload instead of define a default value for `printer` to help
        // debuggers out.
        return annotation_str(nullptr);
    }

    // Returns a human-readable string that includes additional information like
    // "type is inferred rather than explicitly defined" to help construct more
    // user-friendly messages.
    virtual std::string repr_str() const {
        return annotation_str();
    }

    bool is_subtype_of(const Type& other) const {
    }

    // Dynamically cast this object to the subclass indicated by the
    // template variable, returning nullptr if the cast is invalid.
    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastReturnType_t<T>>
    RetType cast() {
        if (T::kind == kind()) {
            return std::static_pointer_cast<T>(static_cast<T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    // cast to SingletonTypePtr<T>
    template<typename T,
             typename = void,
             typename = std::enable_if_t<detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastReturnType_t<T>>
    RetType cast() {
        if (T::kind == kind()) {
            CHECK(this == T::Global().get());
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastConstReturnType_t<T>>
    RetType cast() const {
        if (T::kind == kind()) {
            return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    template<typename T,
             typename = void,
             typename = std::enable_if_t<detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastConstReturnType_t<T>>
    RetType cast() const {
        if (T::kind == kind()) {
            CHECK(this == T::Global().get());
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    T* cast_to_raw_type() {
        if (T::kind == kind()) {
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    const T* cast_to_raw_type() const {
        if (T::kind == kind()) {
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

protected:
    Type() : kind_(TypeKind::AnyType) {}
    explicit Type(TypeKind kind) : kind_(kind) {}
    virtual ~Type() = default;

    virtual std::string annotation_str_impl(const TypePrinter&) const {
        return this->str();
    }

private:
    TypeKind kind_;
};



// Base class for Types that are guaranteed to be owned by std::shared_ptr.
class SharedType : public Type, public std::enable_shared_from_this<SharedType> {
public:
    using Type::Type;
};

template<typename T>
class Singleton : public Type {
public:
    static SingletonTypePtr<T> Global() {
        static T inst;
        return &inst;
    }

    Singleton(const Singleton&) = delete;
    Singleton(Singleton&&) noexcept = delete;
    Singleton& operator=(const Singleton&) = delete;
    Singleton& operator=(Singleton&&) noexcept = delete;

protected:
    explicit Singleton(TypeKind kind) : Type(kind) {}
};

using AnyTypePtr = SingletonTypePtr<AnyType>;
class AnyType : public Singleton<AnyType> {
public:
    std::string str() const override {
        return "Any";
    }

    bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static constexpr auto Kind = TypeKind::AnyType;

private:
    AnyType() : Singleton(TypeKind::AnyType) {}
    friend class Singleton;
};

class NumberType : public Singleton<NumberType> {
public:
    std::string str() const override {
        return "Scalar";
    }

    static constexpr auto Kind = TypeKind::NumberType;

protected:
    NumberType(TypeKind kind = TypeKind::NumberType) : Singleton(kind) {}

private:
    std::string annotation_str_impl(const TypePrinter&) const override {
        return "number";
    }

    friend class Singleton;
};

class IntType : public NumberType {
public:
    std::string str() const override {
        return "int";
    }

    bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static IntType* GetTypePtr() {
        static IntType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::IntType;

private:
    IntType() : NumberType(TypeKind::IntType) {}
    std::string annotation_str_impl(const TypePrinter&) const override {
        return "int";
    }
};

class FloatType : public NumberType {
public:
    std::string str() const override {
        return "float";
    }

    bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    static FloatType* GetTypePtr() {
        static FloatType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::FloatType;

private:
    FloatType() : NumberType(TypeKind::FloatType) {}
    std::string annotation_str_impl(const TypePrinter&) const override {
        return "float";
    }
};

class ComplexType : public NumberType {
public:
    std::string str() const override {
        return "complex";
    }

    bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    static ComplexType* GetTypePtr() {
        static ComplexType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::ComplexType;

private:
    ComplexType() : NumberType(TypeKind::ComplexType) {}
    std::string annotation_str_impl(const TypePrinter&) const override {
        return "complex";
    }
};

class UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;
class UnionType : public SharedType {
public:

    bool isUnionType() const override {
        return true;
    }

    static constexpr auto Kind = TypeKind::UnionType;

protected:
    std::vector<TypePtr> types_;
    bool can_hold_none_;
};

class OptionalType : public UnionType {
public:
    bool isUnionType() const override {
        return true;
    }

    std::string str() const override {
        return get_element_type()->str() + "?";
    }

    const TypePtr& get_element_type() const {
        return containe_type_;
    }

    static constexpr auto Kind = TypeKind::OptionalType;

private:
    TypePtr containe_type_;
};


inline std::string toString(const Type& t) {
    return t.str();
}

inline bool operator==(const Type& lhs, const Type& rhs) {
    return rhs.symmetric() ? lhs.equals(rhs) : rhs.equals(lhs);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return !(lhs == rhs);
}


}// namespace aethermind

#endif//AETHERMIND_TYPE_H
