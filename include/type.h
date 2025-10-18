//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_TYPE_H
#define AETHERMIND_TYPE_H

#include "container/array_view.h"
#include "container/string.h"
#include "data_type.h"
#include "error.h"
#include "tensor.h"
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

    NODISCARD String str() const override {
        return union_str(nullptr, false);
    }

    std::optional<TypePtr> to_optional() const;

    bool canHoldType(const Type& type) const;

    static UnionTypePtr create(const std::vector<TypePtr>& ref);

    static constexpr auto Kind = TypeKind::UnionType;

protected:
    explicit UnionType(const std::vector<TypePtr>& types, TypeKind kind = TypeKind::UnionType);

    String union_str(const TypePrinter& printer = nullptr, bool is_annotation_str = false) const;

    std::vector<TypePtr> types_;
    bool can_hold_none_;
    bool has_free_variables_;
};

class OptionalType;
using OptionalTypePtr = std::shared_ptr<OptionalType>;
class OptionalType : public UnionType {
public:
    bool isUnionType() const override {
        return true;
    }

    NODISCARD String str() const override {
        return get_element_type()->str() + "?";
    }

    bool equals(const Type& rhs) const override;

    bool isSubtypeOfExt(const Type& other, std::ostream* why_not) const override;

    const TypePtr& get_element_type() const {
        return contained_type_;
    }

    static OptionalTypePtr create(const TypePtr& contained);

    static constexpr auto Kind = TypeKind::OptionalType;

private:
    explicit OptionalType(const TypePtr& contained);

    TypePtr contained_type_;
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

struct ShapeSymbol {
    ShapeSymbol() : value_(-1) {}

    NODISCARD int64_t value() const {
        return value_;
    }

    NODISCARD bool is_static() const {
        return value_ >= 0;
    }

    NODISCARD int64_t static_size() const {
        CHECK(is_static());
        return value_;
    }

    bool operator==(const ShapeSymbol& other) const {
        return value_ == other.value_;
    }

    bool operator<(const ShapeSymbol& other) const {
        return value_ < other.value_;
    }

    static ShapeSymbol CreateFromStaticSize(int64_t val) {
        return ShapeSymbol(val);
    }

    static ShapeSymbol Create() {
        return CreateFromStaticSize(-static_cast<int64_t>(++num_symbols_));
    }

private:
    explicit ShapeSymbol(int64_t val) : value_(val) {}

    int64_t value_;
    static std::atomic<size_t> num_symbols_;
};

std::ostream& operator<<(std::ostream& os, const ShapeSymbol& s);

inline ShapeSymbol merge_primitive(const ShapeSymbol& a, const ShapeSymbol& b) {
    if (a.is_static() && b.is_static() && a == b) {
        return a;
    }
    return ShapeSymbol::Create();
}

// Shape of a Tensor represented with ShapeSymbol's. Unranked, ranked unknown
// dims, partially known and fully known shapes are all supported.
struct SymbolicShape {
    SymbolicShape() = default;

    // Known rank but unknown dimensions
    SymbolicShape(std::optional<size_t> rank);//NOLINT

    // Mix of known and unknown ranks
    SymbolicShape(const std::vector<std::optional<int64_t>>& dims);//NOLINT

    SymbolicShape(std::vector<ShapeSymbol> dims) : dims_(std::move(dims)) {}//NOLINT

    SymbolicShape(IntArrayView dims);//NOLINT

    ShapeSymbol operator[](size_t i) const;

    NODISCARD ShapeSymbol at(size_t i) const;

    // Returns rank or nullopt in case of unranked shape.
    NODISCARD std::optional<size_t> rank() const;

    NODISCARD const std::optional<std::vector<ShapeSymbol>>& sizes() const;

    NODISCARD std::optional<std::vector<bool>> symbolic_dims() const;

    // Checks whether the shape is fully defined/complete, i.e. rank and sizes
    // of every dimension are known.
    NODISCARD bool is_complete() const;

    void dump() const;

    // Create new SymbolicShape that is result of merging self and another
    // SymbolicShape. Only dimensions that are static and equal will be
    // preserved.
    // If either of two shapes are of unknown rank or they have unmatching rank,
    // result will be unranked.
    NODISCARD SymbolicShape merge(const SymbolicShape& other) const;

    friend bool operator==(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return lhs.dims_ == rhs.dims_;
    }

    friend bool operator!=(const SymbolicShape& lhs, const SymbolicShape& rhs) {
        return !(lhs == rhs);
    }

private:
    std::optional<std::vector<ShapeSymbol>> dims_{std::nullopt};
};

std::ostream& operator<<(std::ostream& os, const SymbolicShape& s);

struct Stride {
    Stride() = default;

    Stride(const std::optional<size_t>& stride_idx, std::optional<bool> contiguous, const std::optional<size_t>& stride)
        : stride_idx_(stride_idx), contiguous_(contiguous), stride_(stride) {}

    NODISCARD bool is_complete() const {
        return stride_idx_ && contiguous_ && stride_;
    }

    bool operator==(const Stride& other) const {
        return stride_idx_ == other.stride_idx_ &&
               contiguous_ == other.contiguous_ &&
               stride_ == other.stride_;
    }

    std::optional<size_t> stride_idx_;
    std::optional<bool> contiguous_;
    std::optional<size_t> stride_;
};

std::ostream& operator<<(std::ostream& os, const Stride& s);

template<>
inline std::optional<Stride> merge_primitive(const std::optional<Stride>& a, const std::optional<Stride>& b) {
    auto lhs = a;
    auto rhs = b;
    if (!lhs.has_value()) {
        lhs = Stride();
    }

    if (!rhs.has_value()) {
        rhs = Stride();
    }

    auto merged_idx = merge_primitive(lhs->stride_idx_, rhs->stride_idx_);
    auto merged_contiguous = merge_primitive(lhs->contiguous_, rhs->contiguous_);
    auto merged_stride = merge_primitive(lhs->stride_, rhs->stride_);

    if (!(merged_idx.has_value() || merged_contiguous.has_value() || merged_stride.has_value())) {
        return std::optional<Stride>{};
    }

    return Stride(merged_idx, merged_contiguous, merged_stride);
}

namespace details {

inline bool is_complete(const Stride& s) {
    return s.is_complete();
}

template<typename T>
bool is_complete(const T&) {
    return true;
}

}// namespace details

template<typename T>
struct VaryingShape {
    using ListOfOptionalElements = std::vector<std::optional<T>>;

    VaryingShape(ListOfOptionalElements dims) : dims_(std::move(dims)) {}//NOLINT

    VaryingShape(const std::vector<T>& vec)// NOLINT
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    VaryingShape(ArrayView<T> vec)//NOLINT
        : VaryingShape(ListOfOptionalElements(vec.begin(), vec.end())) {}

    VaryingShape(std::optional<size_t> size = std::nullopt) : dims_(std::nullopt) {//NOLINT
        if (size.has_value()) {
            dims_ = ListOfOptionalElements(size.value());
        }
    }

    VaryingShape(size_t size) : VaryingShape(std::optional<size_t>(size)) {}//NOLINT

    const std::optional<T>& operator[](size_t i) const {
        if (!dims_.has_value()) {
            AETHERMIND_THROW(RuntimeError) << "Rank isn't fixed";
        }
        return dims_.value()[i];
    }

    NODISCARD std::optional<size_t> size() const {
        if (!dims_.has_value()) {
            return std::nullopt;
        }

        return dims_.value().size();
    }

    NODISCARD const std::optional<ListOfOptionalElements>& shape() const {
        return dims_;
    }

    bool operator==(const VaryingShape& other) const {
        return dims_ == other.dims_;
    }

    NODISCARD std::optional<std::vector<T>> get_concrete_value() const;

    NODISCARD bool is_complete() const;

    NODISCARD VaryingShape merge(const VaryingShape& other) const;

private:
    std::optional<ListOfOptionalElements> dims_;
};

class TensorType;
using TensorTypePtr = std::shared_ptr<TensorType>;
class TensorType : public SharedType {
public:
    String str() const override {
        return "Tensor";
    }

    String repr_str() const override {
        if (is_inferred_type()) {
            return str() + " (inferred)";
        }
        return str();
    }

    bool equals(const Type& rhs) const override;

    const std::optional<DataType>& data_type() const {
        return dtype_;
    }

    const std::optional<Device>& device() const {
        return device_;
    }

    VaryingShape<int64_t> shape() const;

    VaryingShape<int64_t> strides() const;

    std::optional<size_t> dim() const {
        return shape().size();
    }

    std::optional<size_t> numel() const;

    const VaryingShape<Stride>& stride_properties() const {
        return strides_;
    }

    const std::optional<bool>& requiresGrad() const {
        return requires_grad_;
    }

    bool requires_grad() const override {
        return requires_grad_ ? *requires_grad_ : true;
    }

    const std::optional<bool>& undefined() const {
        return undefined_;
    }

    bool is_inferred_type() const {
        return is_inferred_;
    }

    // is all information about the type specified except for autograd?
    // This replaces the notion of a 'CompleteTensorType' that used to exist
    // in the type-hierarchy. Excluding require_grad and undefined allows
    // this to match the old behavior.
    bool is_complete() const {
        return data_type() && device() && shape_.is_complete() && strides_.is_complete();
    }

    TensorTypePtr contiguous() const;

    static std::vector<int64_t> contiguous_stride_of(IntArrayView shape,
                                                     MemoryFormat memory_format = MemoryFormat::Contiguous);

    static TensorTypePtr create(const Tensor& t);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            const VaryingShape<int64_t>& shape,
            const VaryingShape<int64_t>& strides,
            std::optional<bool> requires_grad,
            std::optional<bool> undefined = false,
            bool tensor_contiguity = false);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            std::optional<size_t> dim,
            std::optional<bool> requires_grad);

    static TensorTypePtr create(
            std::optional<DataType> dtype,
            std::optional<Device> device,
            SymbolicShape shape,
            VaryingShape<Stride> strides,
            std::optional<bool> requires_grad,
            std::optional<bool> undefined = false);

    static TensorTypePtr create_contiguous(DataType dtype, Device device, IntArrayView shape);

    TensorTypePtr with_requires_grad(std::optional<bool> s) const;

    TensorTypePtr with_data_type(const std::optional<DataType>&) const;

    TensorTypePtr with_dim(std::optional<size_t>) const;

    TensorTypePtr with_shape(IntArrayView shape) const;

    TensorTypePtr with_strides(VaryingShape<Stride>) const;

    TensorTypePtr with_device(std::optional<Device> device) const;

    TensorTypePtr with_symbolic_shape(SymbolicShape symbolic_shape) const;

    TensorTypePtr with_shape_and_strides(IntArrayView shape, IntArrayView strides) const;

    TensorTypePtr with_undefined() const;

    TensorTypePtr with_possibly_undefined() const;

    static constexpr auto Kind = TypeKind::TensorType;

private:
    // constructor
    TensorType(std::optional<DataType> dtype,
               std::optional<Device> device,
               SymbolicShape shape,
               VaryingShape<Stride> strides,
               std::optional<bool> requires_grad,
               std::optional<bool> undefined = false);

    static VaryingShape<Stride> compute_stride_props(
            IntArrayView shape,
            IntArrayView strides,
            bool tensor_contiguity = false);

    TensorTypePtr clone() const {
        auto ptr = new TensorType(dtype_, device_, shape_, strides_, requires_grad_, undefined_);
        return TensorTypePtr(ptr);
    }

    std::optional<DataType> dtype_;
    std::optional<Device> device_;
    SymbolicShape shape_;
    VaryingShape<Stride> strides_;
    std::optional<bool> requires_grad_;
    std::optional<bool> undefined_;
    // Whether this type was inferred.
    bool is_inferred_ = false;
};

template<typename T>
std::ostream& operator<<(std::ostream& os, const VaryingShape<T>& t);

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

bool is_contiguous_stride(IntArrayView shape, IntArrayView strides);

}// namespace aethermind

#endif//AETHERMIND_TYPE_H
