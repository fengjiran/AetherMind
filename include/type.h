//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_H
#define AETHERMIND_TYPE_H

#include "container/array_view.h"
#include "error.h"
#include "macros.h"
#include "type.h"
#include "type_ptr.h"

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
DECLARE_SINGLETON_TYPE(NoneType);
DECLARE_SINGLETON_TYPE(NumberType);
DECLARE_SINGLETON_TYPE(IntType);
DECLARE_SINGLETON_TYPE(FloatType);
DECLARE_SINGLETON_TYPE(ComplexType);
DECLARE_SINGLETON_TYPE(StringType);
DECLARE_SINGLETON_TYPE(DeviceObjType);

using TypePtr = SingletonOrSharedTypePtr<Type>;

class Type {
public:
    NODISCARD TypeKind kind() const {
        return kind_;
    }

    NODISCARD virtual std::string str() const = 0;

    // a == b
    NODISCARD virtual bool equals(const Type& rhs) const = 0;

    // a == b <=> b == a
    NODISCARD virtual bool symmetric() const {
        return true;
    }

    NODISCARD virtual bool isUnionType() const {
        return false;
    }

    // list of types this type contains, e.g. for a List then element type of
    // list for a tuple, the types of the tuple elements
    NODISCARD virtual ArrayView<TypePtr> containedTypes() const {
        return {};
    }

    NODISCARD virtual TypePtr containedType(size_t i) const {
        return containedTypes().at(i);
    }

    NODISCARD virtual size_t containedTypeSize() const {
        return containedTypes().size();
    }

    NODISCARD virtual bool hasFreeVars() const {
        return false;
    }

    NODISCARD std::string annotation_str(const TypePrinter& printer) const {
        if (printer) {
            if (auto renamed = printer(*this)) {
                return *renamed;
            }
        }
        return this->annotation_str_impl(printer);
    }

    NODISCARD std::string annotation_str() const {
        // Overload instead of define a default value for `printer` to help
        // debuggers out.
        return annotation_str(nullptr);
    }

    // Returns a human-readable string that includes additional information like
    // "type is inferred rather than explicitly defined" to help construct more
    // user-friendly messages.
    NODISCARD virtual std::string repr_str() const {
        return annotation_str();
    }

    virtual bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const;

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool isSubTypeOfExt(const SingletonOrSharedTypePtr<T>& other, std::ostream* why_not) const {
        return isSubtypeOfExt(*other, why_not);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool isSubTypeOfExt(const std::shared_ptr<T>& other, std::ostream* why_not) const {
        return isSubtypeOfExt(*other, why_not);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool isSubTypeOfExt(const SingletonTypePtr<T>& other, std::ostream* why_not) const {
        return isSubtypeOfExt(*other, why_not);
    }

    NODISCARD bool is_subtype_of(const Type& other) const {
        return isSubtypeOfExt(other, nullptr);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool is_subtype_of(const SingletonOrSharedTypePtr<T>& other) const {
        return is_subtype_of(*other);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool is_subtype_of(const std::shared_ptr<T>& other) const {
        return is_subtype_of(*other);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool is_subtype_of(const SingletonTypePtr<T>& other) const {
        return is_subtype_of(*other);
    }

    NODISCARD virtual bool is_module() const {
        return false;
    }

    NODISCARD virtual bool hasFreeVariables() const {
        return false;
    }

    // Dynamically cast this object to the subclass indicated by the
    // template variable, returning nullptr if the cast is invalid.
    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastReturnType_t<T>>
    RetType cast() {
        if (T::Kind == kind()) {
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
        if (T::Kind == kind()) {
            CHECK(this == T::Global().get());
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastConstReturnType_t<T>>
    RetType cast() const {
        if (T::Kind == kind()) {
            return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    template<typename T,
             typename = void,
             typename = std::enable_if_t<detail::is_singleton_type_v<T>>,
             typename RetType = detail::CastConstReturnType_t<T>>
    RetType cast() const {
        if (T::Kind == kind()) {
            CHECK(this == T::Global().get());
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    T* cast_to_raw_type() {
        if (T::Kind == kind()) {
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    const T* cast_to_raw_type() const {
        if (T::Kind == kind()) {
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    auto expect() {
        auto r = cast<T>();
        CHECK(r);
        return r;
    }

    template<typename T>
    auto expect() const {
        auto r = cast<const T>();
        CHECK(r);
        return r;
    }

    template<typename T>
    T& expectRef() {
        auto* r = cast_to_raw_type<T>();
        CHECK(r);
        return *r;
    }

    template<typename T>
    const T& expectRef() const {
        auto* r = cast_to_raw_type<const T>();
        CHECK(r);
        return *r;
    }

protected:
    explicit Type(TypeKind kind) : kind_(kind) {}
    virtual ~Type() = default;

    NODISCARD virtual std::string annotation_str_impl(const TypePrinter&) const {
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

// common base for all types that have a single sub element
// e.g. Future[T], Optional[T], List[T]
template<typename T, TypeKind K>
class SingleElementType : public SharedType {
public:
    const TypePtr& get_element_type() const {
        return elem_;
    }

    NODISCARD bool hasFreeVars() const override {
        return get_element_type()->hasFreeVars();
    }

    ArrayView<TypePtr> containedTypes() const override {
        return elem_;
    }

    bool equals(const Type& rhs) const override {
        if (auto cast_rhs = rhs.cast<T>()) {
            return *get_element_type() == *cast_rhs->get_element_type();
        }
        return false;
    }

    static constexpr auto Kind = K;

protected:
    explicit SingleElementType(TypePtr elem) : SharedType(Kind), elem_(std::move(elem)) {
        if (!elem_) {
            AETHERMIND_THROW(runtime_error) << "Cannot create " << TypeKindToString(Kind) << " with none type.";
        }
    }

private:
    TypePtr elem_;
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
    NODISCARD std::string str() const override {
        return "Any";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static constexpr auto Kind = TypeKind::AnyType;

private:
    AnyType() : Singleton(Kind) {}
    friend class Singleton;
};

using NoneTypePtr = SingletonTypePtr<NoneType>;
class NoneType : public Singleton<NoneType> {
public:
    NODISCARD bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    NODISCARD std::string str() const override {
        return "None";
    }

    static constexpr auto Kind = TypeKind::NoneType;

private:
    NoneType() : Singleton(Kind) {}
    friend class Singleton;
};

using NumberTypePtr = SingletonTypePtr<NumberType>;
class NumberType : public Singleton<NumberType> {
public:
    NODISCARD std::string str() const override {
        return "Scalar";
    }

    NODISCARD bool equals(const Type& other) const override;

    NODISCARD bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override;

    static constexpr auto Kind = TypeKind::NumberType;

protected:
    explicit NumberType(TypeKind kind = Kind) : Singleton(kind) {}

private:
    NODISCARD std::string annotation_str_impl(const TypePrinter&) const override {
        return "number";
    }

    friend class Singleton;
};

using IntTypePtr = SingletonTypePtr<IntType>;
class IntType : public NumberType {
public:
    NODISCARD std::string str() const override {
        return "int";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    NODISCARD bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override {
        return other.kind() == NumberType::Kind || NumberType::isSubtypeOfExt(other, why_not);
    }

    static IntTypePtr Global() {
        static IntType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::IntType;

private:
    IntType() : NumberType(Kind) {}
    NODISCARD std::string annotation_str_impl(const TypePrinter&) const override {
        return "int";
    }
};

using FloatTypePtr = SingletonTypePtr<FloatType>;
class FloatType : public NumberType {
public:
    NODISCARD std::string str() const override {
        return "float";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    NODISCARD bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override {
        return other.kind() == NumberType::Kind || NumberType::isSubtypeOfExt(other, why_not);
    }

    static FloatTypePtr Global() {
        static FloatType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::FloatType;

private:
    FloatType() : NumberType(Kind) {}
    NODISCARD std::string annotation_str_impl(const TypePrinter&) const override {
        return "float";
    }
};

using ComplexTypePtr = SingletonTypePtr<ComplexType>;
class ComplexType : public NumberType {
public:
    NODISCARD std::string str() const override {
        return "complex";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    NODISCARD bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override {
        return other.kind() == NumberType::Kind || NumberType::isSubtypeOfExt(other, why_not);
    }

    static ComplexTypePtr Global() {
        static ComplexType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::ComplexType;

private:
    ComplexType() : NumberType(Kind) {}
    // NODISCARD std::string annotation_str_impl(const TypePrinter&) const override {
    //     return "complex";
    // }
};

using StringTypePtr = SingletonOrSharedTypePtr<StringType>;
class StringType : public Singleton<StringType> {
public:
    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    NODISCARD std::string str() const override {
        return "str";
    }

    static constexpr auto Kind = TypeKind::StringType;

private:
    explicit StringType() : Singleton(Kind) {}
    friend class Singleton;
};

using DeviceObjTypePtr = SingletonOrSharedTypePtr<DeviceObjType>;
class DeviceObjType : public Singleton<DeviceObjType> {
public:
    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    NODISCARD std::string str() const override {
        return "Device";
    }

    static constexpr auto Kind = TypeKind::DeviceObjType;

private:
    explicit DeviceObjType() : Singleton(Kind) {}
    friend class Singleton;
};


class UnionType;
using UnionTypePtr = std::shared_ptr<UnionType>;
class UnionType : public SharedType {
public:
    bool isUnionType() const override {
        return true;
    }

    bool equals(const Type& rhs) const override;

    bool isSubtypeOfExt(const Type& rhs, std::ostream* why_not) const override;

    ArrayView<TypePtr> containedTypes() const override {
        return types_;
    }

    bool hasFreeVariables() const override {
        return has_free_variables_;
    }

    // just for test
    ArrayView<TypePtr> getTypes() const {
        return types_;
    }

    std::string str() const override {
        return union_str(nullptr, false);
    }

    std::optional<TypePtr> to_optional() const;

    bool canHoldType(const Type& type) const;

    static UnionTypePtr create(const std::vector<TypePtr>& ref);

    static constexpr auto Kind = TypeKind::UnionType;

protected:
    explicit UnionType(const std::vector<TypePtr>& types, TypeKind kind = TypeKind::UnionType);

    std::string union_str(const TypePrinter& printer = nullptr, bool is_annotation_str = false) const;

    std::vector<TypePtr> types_;
    bool can_hold_none_;
    bool has_free_variables_;

    // friend class Type;
};

class OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
class OptionalType : public UnionType {
public:
    bool isUnionType() const override {
        return true;
    }

    std::string str() const override {
        return get_element_type()->str() + "?";
    }

    bool equals(const Type& rhs) const override;

    bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override;

    const TypePtr& get_element_type() const {
        return containe_type_;
    }

    static OptionalTypePtr create(const TypePtr& contained);

    static constexpr auto Kind = TypeKind::OptionalType;

private:
    explicit OptionalType(const TypePtr& contained);

    TypePtr containe_type_;

    // friend class Type;
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


// Attempt to find the correct supertype of the two types `t1` and `t2`.
// If no supertype is found, then nullopt will be returned if
// `default_to_union` is false, and `Union[t1, t2]` will be returned
// if it is true. If `t1 == t2`, or `t1` is a type refinement of `t2`,
// then `t2` will be returned (and vice versa).
//
// Two different tensortypes will return dynamic.
//
// Currently, we chose not to support returning a NumberType for
// two types from the set of {FloatType, IntType, ComplexType}, because
// there is a lack of operator support for NumberType.
//
// If `type_hint` is an `InterfaceType`, then we can use that as a
// potential supertype for `ClassType`s in the list. Otherwise, we have
// no way to find and use some common interface type
std::optional<TypePtr> unify_types(const TypePtr& t1, const TypePtr& t2,
                                   bool default_to_union = false,
                                   const TypePtr& type_hint = nullptr);

void StandardizeVectorForUnion(const std::vector<TypePtr>& ref, std::vector<TypePtr>& need_to_fill);
void StandardizeVectorForUnion(std::vector<TypePtr>& to_flatten);
}// namespace aethermind

#endif//AETHERMIND_TYPE_H
