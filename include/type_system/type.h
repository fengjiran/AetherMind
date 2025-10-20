//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_SYSTEM_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_TYPE_H

#include "container/array_view.h"
#include "container/string.h"
#include "data_type.h"
#include "error.h"
#include "type.h"
#include "type_ptr.h"

#include <functional>

namespace aethermind {

#define AETHERMIND_ALL_TYPES(_) \
    _(AnyType)                  \
    _(EnumType)                 \
    _(AnyEnumType)              \
    _(TensorType)               \
    _(StorageType)              \
    _(TupleType)                \
    _(ListType)                 \
    _(DictType)                 \
    _(NumberType)               \
    _(FloatType)                \
    _(ComplexType)              \
    _(FutureType)               \
    _(AwaitType)                \
    _(RRefType)                 \
    _(IntType)                  \
    _(NoneType)                 \
    _(StringType)               \
    _(GeneratorType)            \
    _(QuantizerType)            \
    _(BoolType)                 \
    _(OptionalType)             \
    _(VarType)                  \
    _(DeviceObjType)            \
    _(StreamObjType)            \
    _(FunctionType)             \
    _(ClassType)                \
    _(PyObjectType)             \
    _(CapsuleType)              \
    _(InterfaceType)            \
    _(QSchemeType)              \
    _(ScalarTypeType)           \
    _(LayoutType)               \
    _(MemoryFormatType)         \
    _(AnyListType)              \
    _(AnyTupleType)             \
    _(AnyClassType)             \
    _(SymIntType)               \
    _(SymFloatType)             \
    _(SymBoolType)              \
    _(UnionType)                \
    _(DynamicType)

enum class TypeKind {
#define DEFINE_TYPE(T) T,
    AETHERMIND_ALL_TYPES(DEFINE_TYPE)
#undef DEFINE_TYPE
};

String TypeKindToString(TypeKind kind);

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
DECLARE_SINGLETON_TYPE(BoolType);
DECLARE_SINGLETON_TYPE(ComplexType);
DECLARE_SINGLETON_TYPE(StringType);
DECLARE_SINGLETON_TYPE(DeviceObjType);

using TypePtr = SingletonOrSharedTypePtr<Type>;

class Type {
public:
    NODISCARD TypeKind kind() const {
        return kind_;
    }

    NODISCARD virtual String str() const = 0;

    // a == b
    NODISCARD virtual bool equals(const Type& rhs) const = 0;

    // a == b <=> b == a
    NODISCARD virtual bool symmetric() const {
        return true;
    }

    NODISCARD virtual bool isUnionType() const {
        return false;
    }

    NODISCARD virtual bool requires_grad() const {
        for (const auto& ct: containedTypes()) {
            if (ct->requires_grad()) {
                return true;
            }
        }
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

    NODISCARD String annotation_str(const TypePrinter& printer) const {
        if (printer) {
            if (auto renamed = printer(*this)) {
                return *renamed;
            }
        }
        return this->annotation_str_impl(printer);
    }

    NODISCARD String annotation_str() const {
        // Overload instead of define a default value for `printer` to help
        // debuggers out.
        return annotation_str(nullptr);
    }

    // Returns a human-readable string that includes additional information like
    // "type is inferred rather than explicitly defined" to help construct more
    // user-friendly messages.
    NODISCARD virtual String repr_str() const {
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
    // cast to SharedTypePtr<T>
    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>>
    auto cast() -> detail::CastReturnType_t<T> {
        if (T::Kind == kind()) {
            return std::static_pointer_cast<T>(static_cast<T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    // cast to SingletonTypePtr<T>
    template<typename T,
             typename = void,
             typename = std::enable_if_t<detail::is_singleton_type_v<T>>>
    auto cast() -> detail::CastReturnType_t<T> {
        if (T::Kind == kind()) {
            CHECK(this == T::Global().get());
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T,
             typename = std::enable_if_t<!detail::is_singleton_type_v<T>>>
    auto cast() const -> detail::CastConstReturnType_t<T> {
        if (T::Kind == kind()) {
            return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    template<typename T,
             typename = void,
             typename = std::enable_if_t<detail::is_singleton_type_v<T>>>
    auto cast() const -> detail::CastConstReturnType_t<T> {
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

    NODISCARD virtual String annotation_str_impl(const TypePrinter&) const {
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
        return ArrayView<TypePtr>(elem_);
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
    NODISCARD String str() const override {
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

    NODISCARD String str() const override {
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
    NODISCARD String str() const override {
        return "Scalar";
    }

    NODISCARD bool equals(const Type& other) const override;

    NODISCARD bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override;

    static constexpr auto Kind = TypeKind::NumberType;

protected:
    explicit NumberType(TypeKind kind = Kind) : Singleton(kind) {}

private:
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "number";
    }

    friend class Singleton;
};

using IntTypePtr = SingletonTypePtr<IntType>;
class IntType : public NumberType {
public:
    NODISCARD String str() const override {
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
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "int";
    }
};

using FloatTypePtr = SingletonTypePtr<FloatType>;
class FloatType : public NumberType {
public:
    NODISCARD String str() const override {
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
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "float";
    }
};

using BoolTypePtr = SingletonTypePtr<BoolType>;
class BoolType : public Singleton<BoolType> {
public:
    NODISCARD String str() const override {
        return "bool";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    static constexpr auto Kind = TypeKind::BoolType;

private:
    BoolType() : Singleton(Kind) {}
};

using ComplexTypePtr = SingletonTypePtr<ComplexType>;
class ComplexType : public NumberType {
public:
    NODISCARD String str() const override {
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
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "complex";
    }
};

using StringTypePtr = SingletonOrSharedTypePtr<StringType>;
class StringType : public Singleton<StringType> {
public:
    NODISCARD String str() const override {
        return "string";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static constexpr auto Kind = TypeKind::StringType;

private:
    explicit StringType() : Singleton(Kind) {}
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "string";
    }
    friend class Singleton;
};

using DeviceObjTypePtr = SingletonOrSharedTypePtr<DeviceObjType>;
class DeviceObjType : public Singleton<DeviceObjType> {
public:
    NODISCARD String str() const override {
        return "Device";
    }

    NODISCARD bool equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }
    static constexpr auto Kind = TypeKind::DeviceObjType;

private:
    explicit DeviceObjType() : Singleton(Kind) {}
    NODISCARD String annotation_str_impl(const TypePrinter&) const override {
        return "Device";
    }
    friend class Singleton;
};

inline String toString(const Type& t) {
    return t.str();
}

inline String toString(const TypePtr& p) {
    return toString(*p);
}

inline bool operator==(const Type& lhs, const Type& rhs) {
    return rhs.symmetric() ? lhs.equals(rhs) : rhs.equals(lhs);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return !(lhs == rhs);
}

template<typename T>
std::optional<T> merge_primitive(const std::optional<T>& a, const std::optional<T>& b) {
    if (a.has_value() && b.has_value() && a.value() == b.value()) {
        return a;
    }
    return std::nullopt;
}

namespace details {

template<typename T>
bool is_complete(const T&) {
    return true;
}

}// namespace details

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

bool is_contiguous_stride(IntArrayView shape, IntArrayView strides);

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_TYPE_H
