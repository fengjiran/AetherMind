//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_SYSTEM_TYPE_H
#define AETHERMIND_TYPE_SYSTEM_TYPE_H

#include "container/array_view.h"
#include "container/string.h"
#include "data_type.h"
#include "error.h"
#include "function.h"
#include "type_ptr.h"
#include "utils/qualified_name.h"

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
using TypePrinter = std::function<std::optional<String>(const Type&)>;

namespace details {

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

}// namespace details


#define DECLARE_SINGLETON_TYPE(Type)                  \
    class Type;                                       \
    namespace details {                               \
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
    AM_NODISCARD TypeKind kind() const;

    AM_NODISCARD virtual String str() const = 0;

    // a == b
    AM_NODISCARD virtual bool Equals(const Type& rhs) const = 0;

    // a == b <=> b == a
    AM_NODISCARD virtual bool IsSymmetric() const;

    AM_NODISCARD virtual bool IsUnionType() const;

    AM_NODISCARD virtual bool requires_grad() const;

    // list of types this type contains, e.g. for a List then element type of
    // list for a tuple, the types of the tuple elements
    AM_NODISCARD virtual ArrayView<TypePtr> GetContainedTypes() const;

    AM_NODISCARD virtual TypePtr GetContainedType(size_t i) const;

    AM_NODISCARD virtual size_t GetContainedTypeSize() const;

    AM_NODISCARD virtual TypePtr CreateWithContainedTypes(const std::vector<TypePtr>&) const;

    // create a new version of this type, replacing its contained types with
    // contained_types
    TypePtr WithContainedTypes(const std::vector<TypePtr>& contained_types);

    AM_NODISCARD virtual bool HasFreeVars() const;

    // Returns a human-readable string that includes additional information like
    // "type is inferred rather than explicitly defined" to help construct more
    // user-friendly messages.
    AM_NODISCARD virtual String ReprStr() const;

    AM_NODISCARD virtual bool IsModule() const;

    AM_NODISCARD String Annotation(const TypePrinter& printer) const;
    AM_NODISCARD String Annotation() const;

    AM_NODISCARD virtual bool IsSubtypeOfImpl(const Type& other) const;

    AM_NODISCARD bool IsSubtypeOf(const Type& other) const;

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool IsSubtypeOf(const std::shared_ptr<T>& other) const {
        return IsSubtypeOf(*other);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool IsSubtypeOf(const SingletonTypePtr<T>& other) const {
        return IsSubtypeOf(*other);
    }

    template<typename T, typename = std::enable_if_t<std::is_base_of_v<Type, T>>>
    bool IsSubtypeOf(const SingletonOrSharedTypePtr<T>& other) const {
        return IsSubtypeOf(*other);
    }

    // Dynamically cast this object to the subclass indicated by the
    // template variable, return nullptr if the cast is invalid.
    // cast to std::shared_ptr<T>
    template<typename T, std::enable_if_t<!details::is_singleton_type_v<T>>* = nullptr>
    std::shared_ptr<T> Cast() {
        if (T::Kind == kind()) {
            return std::static_pointer_cast<T>(static_cast<T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    // cast to SingletonTypePtr<T>
    template<typename T, std::enable_if_t<details::is_singleton_type_v<T>>* = nullptr>
    SingletonTypePtr<T> Cast() {
        if (T::Kind == kind()) {
            AM_CHECK(this == T::Global().get());
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    // cast to std::shared_ptr<const T>
    template<typename T, std::enable_if_t<!details::is_singleton_type_v<T>>* = nullptr>
    AM_NODISCARD std::shared_ptr<const T> Cast() const {
        if (T::Kind == kind()) {
            return std::static_pointer_cast<const T>(static_cast<const T*>(this)->shared_from_this());
        }
        return nullptr;
    }

    // cast to SingletonTypePtr<const T>
    template<typename T, std::enable_if_t<details::is_singleton_type_v<T>>* = nullptr>
    SingletonTypePtr<const T> Cast() const {
        if (T::Kind == kind()) {
            AM_CHECK(this == T::Global().get());
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    T* CastToRawTypePtr() {
        if (T::Kind == kind()) {
            return static_cast<T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    const T* CastToRawTypePtr() const {
        if (T::Kind == kind()) {
            return static_cast<const T*>(this);
        }
        return nullptr;
    }

    template<typename T>
    decltype(auto) Expect() {
        auto r = Cast<T>();
        AM_CHECK(r);
        return r;
    }

    template<typename T>
    decltype(auto) Expect() const {
        auto r = Cast<const T>();
        AM_CHECK(r);
        return r;
    }

    template<typename T>
    T& ExpectRef() {
        auto* r = CastToRawTypePtr<T>();
        AM_CHECK(r != nullptr);
        return *r;
    }

    template<typename T>
    const T& ExpectRef() const {
        auto* r = CastToRawTypePtr<const T>();
        AM_CHECK(r != nullptr);
        return *r;
    }

protected:
    explicit Type(TypeKind kind) : kind_(kind) {}

    virtual ~Type() = default;

    AM_NODISCARD virtual String AnnotationImpl(const TypePrinter&) const;

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
    const TypePtr& GetElementType() const {
        return elem_;
    }

    AM_NODISCARD bool HasFreeVars() const override {
        return GetElementType()->HasFreeVars();
    }

    ArrayView<TypePtr> GetContainedTypes() const override {
        return ArrayView(elem_);
    }

    bool Equals(const Type& rhs) const override {
        if (auto cast_rhs = rhs.Cast<T>()) {
            return *GetElementType() == *cast_rhs->GetElementType();
        }
        return false;
    }

    static constexpr auto Kind = K;

protected:
    explicit SingleElementType(TypePtr elem) : SharedType(Kind), elem_(std::move(elem)) {
        if (!elem_) {
            AM_THROW(runtime_error) << "Cannot create " << TypeKindToString(Kind)
                                            << " with none type.";
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
    AM_NODISCARD String str() const override {
        return "Any";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
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
    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    AM_NODISCARD String str() const override {
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
    AM_NODISCARD String str() const override {
        return "Scalar";
    }

    AM_NODISCARD bool Equals(const Type& other) const override;

    AM_NODISCARD bool IsSubtypeOfImpl(const Type& other) const override;

    static constexpr auto Kind = TypeKind::NumberType;

protected:
    explicit NumberType(TypeKind kind = Kind) : Singleton(kind) {}

private:
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "number";
    }

    friend class Singleton;
};

using IntTypePtr = SingletonTypePtr<IntType>;
class IntType : public NumberType {
public:
    AM_NODISCARD String str() const override {
        return "int";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    AM_NODISCARD bool IsSubtypeOfImpl(const Type& other) const override {
        return other.kind() == NumberType::Kind || NumberType::IsSubtypeOfImpl(other);
    }

    static IntTypePtr Global() {
        static IntType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::IntType;

private:
    IntType() : NumberType(Kind) {}
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "int";
    }
};

using FloatTypePtr = SingletonTypePtr<FloatType>;
class FloatType : public NumberType {
public:
    AM_NODISCARD String str() const override {
        return "float";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    AM_NODISCARD bool IsSubtypeOfImpl(const Type& other) const override {
        return other.kind() == NumberType::Kind || NumberType::IsSubtypeOfImpl(other);
    }

    static FloatTypePtr Global() {
        static FloatType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::FloatType;

private:
    FloatType() : NumberType(Kind) {}
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "float";
    }
};

using ComplexTypePtr = SingletonTypePtr<ComplexType>;
class ComplexType : public NumberType {
public:
    AM_NODISCARD String str() const override {
        return "complex";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    AM_NODISCARD bool IsSubtypeOfImpl(const Type& other) const override {
        return other.kind() == NumberType::Kind || NumberType::IsSubtypeOfImpl(other);
    }

    static ComplexTypePtr Global() {
        static ComplexType inst;
        return &inst;
    }

    static constexpr auto Kind = TypeKind::ComplexType;

private:
    ComplexType() : NumberType(Kind) {}
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "complex";
    }
};

using BoolTypePtr = SingletonTypePtr<BoolType>;
class BoolType : public Singleton<BoolType> {
public:
    AM_NODISCARD String str() const override {
        return "bool";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return rhs.kind() == kind();
    }

    static constexpr auto Kind = TypeKind::BoolType;

private:
    BoolType() : Singleton(Kind) {}

    friend class Singleton;
};

using StringTypePtr = SingletonOrSharedTypePtr<StringType>;
class StringType : public Singleton<StringType> {
public:
    AM_NODISCARD String str() const override {
        return "string";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static constexpr auto Kind = TypeKind::StringType;

private:
    explicit StringType() : Singleton(Kind) {}
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "string";
    }
    friend class Singleton;
};

using DeviceObjTypePtr = SingletonOrSharedTypePtr<DeviceObjType>;
class DeviceObjType : public Singleton<DeviceObjType> {
public:
    AM_NODISCARD String str() const override {
        return "Device";
    }

    AM_NODISCARD bool Equals(const Type& rhs) const override {
        return kind() == rhs.kind();
    }

    static constexpr auto Kind = TypeKind::DeviceObjType;

private:
    explicit DeviceObjType() : Singleton(Kind) {}
    AM_NODISCARD String AnnotationImpl(const TypePrinter&) const override {
        return "Device";
    }
    friend class Singleton;
};

class NamedType;
using NamedTypePtr = std::shared_ptr<NamedType>;
using ConstNamedTypePtr = std::shared_ptr<const NamedType>;
class NamedType : public SharedType {
public:
    NamedType(TypeKind kind, std::optional<QualifiedName> name) : SharedType(kind), name_(std::move(name)) {
        AM_CHECK(kind == TypeKind::TupleType || kind == TypeKind::FunctionType ||
                         kind == TypeKind::ClassType || kind == TypeKind::InterfaceType ||
                         kind == TypeKind::EnumType,
                 "If you add a new kind of NamedType, please update the cast<NamedType> specialization and this assert");
    }

    // Fully qualified name of type
    // Looks like: "foo.bar.Baz".
    const std::optional<QualifiedName>& name() const {
        return name_;
    }

private:
    std::optional<QualifiedName> name_;
};

class FunctionType;
using FunctionTypePtr = std::shared_ptr<FunctionType>;
class FunctionType : public NamedType {
public:
    bool Equals(const Type& rhs) const override {
        if (auto func_type = rhs.Cast<FunctionType>()) {
            return func_type->function_.get_impl_ptr_unsafe() == function_.get_impl_ptr_unsafe();
        }
        return false;
    }

    String str() const override {
        return "Function";
    }

    Function function() const {
        return function_;
    }

    static FunctionTypePtr Create(const Function& function) {
        return FunctionTypePtr(new FunctionType(function));//NOLINT
    }

    static constexpr auto Kind = TypeKind::FunctionType;

private:
    FunctionType(const Function& function)// NOLINT
        : NamedType(TypeKind::FunctionType, function.GetQualifiedName()), function_(function) {}

    String AnnotationImpl(const TypePrinter&) const override {
        return name()->GetQualifiedName();
    }

    Function function_;
};

inline String toString(const Type& t) {
    return t.str();
}

inline String toString(const TypePtr& p) {
    return toString(*p);
}

inline bool operator==(const Type& lhs, const Type& rhs) {
    return rhs.IsSymmetric() ? lhs.Equals(rhs) : rhs.Equals(lhs);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return !(lhs == rhs);
}

template<typename T>
std::optional<T> MergePrimitiveValue(const std::optional<T>& a, const std::optional<T>& b) {
    if (a.has_value() && b.has_value() && a.value() == b.value()) {
        return a;
    }
    return std::nullopt;
}

namespace details {

template<typename T>
bool IsComplete(const T&) {
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

bool IsContiguousStride(IntArrayView shape, IntArrayView strides);
bool IsCrossDimensionOverlap(IntArrayView shape, IntArrayView strides);

}// namespace aethermind

#endif//AETHERMIND_TYPE_SYSTEM_TYPE_H
