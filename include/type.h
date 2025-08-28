//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_H
#define AETHERMIND_TYPE_H

#include "macros.h"

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

// The wrapper for singleton type pointer, as the tag of singleton type.
template<typename T>
class SingletonTypePtr {
public:
    using element_type = std::remove_extent_t<T>;

    SingletonTypePtr(T* ptr) : ptr_(ptr) {}// NOLINT

    explicit SingletonTypePtr(std::shared_ptr<T>) = delete;

    T* get() const {
        return ptr_;
    }

    T& operator*() const {
        return *ptr_;
    }

    T* operator->() const {
        return get();
    }

    operator bool() const {
        return get() != nullptr;
    }

private:
    T* ptr_{nullptr};
};

template<typename T, typename U>
bool operator==(const SingletonTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return static_cast<void*>(lhs.get()) == static_cast<void*>(rhs.get());
}

template<typename T, typename U>
bool operator!=(const SingletonTypePtr<T>& lhs, const SingletonTypePtr<U>& rhs) {
    return !(lhs == rhs);
}

const char* TypeKindToString(TypeKind kind);

class Type;

// Use this to customize how a Type is printed using `annotation_str()`. If
// std::nullopt is returned, `annotation_str()` falls through to its default
// implementation.
using TypePrinter = std::function<std::optional<std::string>(const Type&)>;

template<typename T>
struct IsSingletonType : std::false_type {};

template<typename T>
using is_singleton_type_v = IsSingletonType<T>::value;

#define DECLARE_SINGLETON_TYPE(Type) \
    template<>                       \
    struct IsSingletonType<Type> : std::true_type {};

template<typename T>
class SingletonOrSharedTypePtr {
public:
private:
    struct SharedPtrWrapper {
        std::shared_ptr<T> ptr_;

        SharedPtrWrapper(std::shared_ptr<T>&& ptr) : ptr_(std::move(ptr)) {}
    };

    struct SingletonRepr {
        T* singleton_;
        void* unused_{nullptr};

        explicit SingletonRepr(T* singleton) : singleton_(singleton) {}
    };

    struct RawRepr {
        void* first_;
        void* null_if_singleton_;
    };

    union Repr {
        Repr() : Repr(nullptr) {}

        explicit Repr(std::nullptr_t) : singleton_repr_(nullptr) {}

        explicit Repr(std::shared_ptr<T> ptr) : shared_repr_(std::move(ptr)) {}

        explicit Repr(SingletonTypePtr<T> ptr) : singleton_repr_(ptr.get()) {}

        SharedPtrWrapper shared_repr_;
        SingletonRepr singleton_repr_;
    };

    Repr repr_;
};

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
    virtual std::vector<Type*> containedTypes() const {
        return {};
    }

    virtual Type* containedType(size_t i) const {
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

template<typename T>
class Singleton : public Type {
public:
    static T* GetTypePtr() {
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


inline std::string toString(const Type& t) {
    return t.str();
}

inline bool operator==(const Type& lhs, const Type& rhs) {
    return rhs.symmetric() ? lhs.equals(rhs) : rhs.equals(lhs);
}

inline bool operator!=(const Type& lhs, const Type& rhs) {
    return !(lhs == rhs);
}

DECLARE_SINGLETON_TYPE(AnyType);
DECLARE_SINGLETON_TYPE(NumberType);
DECLARE_SINGLETON_TYPE(IntType);
DECLARE_SINGLETON_TYPE(FloatType);

}// namespace aethermind

#endif//AETHERMIND_TYPE_H
