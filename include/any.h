//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "any_utils.h"
#include "data_type.h"
#include "device.h"
#include "error.h"

#include <typeindex>
#include <variant>

namespace aethermind {

#ifdef CRTP_HOLDER

template<typename Derived>
class HolderBase_ {
public:
    NODISCARD std::unique_ptr<HolderBase_> Clone() const {
        return static_cast<const Derived*>(this)->CloneImpl();
    }

    NODISCARD const std::type_index& type() const {
        return static_cast<const Derived*>(this)->type_impl();
    }

    NODISCARD uint32_t use_count() const {
        return static_cast<const Derived*>(this)->use_count_impl();
    }

    NODISCARD bool IsObjectRef() const {
        return static_cast<const Derived*>(this)->IsObjectRefImpl();
    }

    NODISCARD bool IsMap() const {
        return static_cast<const Derived*>(this)->IsMapImpl();
    }

    NODISCARD void* GetUnderlyingPtr() const {
        return static_cast<const Derived*>(this)->GetUnderlyingPtrImpl();
    }
};

template<typename T>
class Holder_ final : public HolderBase_<Holder_<T>> {
public:
    explicit Holder_(T value) : value_(std::move(value)), type_index_(typeid(T)) {}

    NODISCARD std::unique_ptr<HolderBase_<Holder_>> CloneImpl() const {
        return std::make_unique<Holder_>(value_);
    }

    NODISCARD const std::type_index& type_impl() const {
        return type_index_;
    }

    NODISCARD uint32_t use_count_impl() const {
        if constexpr (details::has_use_count_method_v<T>) {
            return value_.use_count();
        } else {
            return 1;
        }
    }

    NODISCARD bool IsObjectRefImpl() const {
        if constexpr (std::is_base_of_v<ObjectRef, T>) {
            return true;
        } else {
            return false;
        }
    }

    NODISCARD bool IsMapImpl() const {
        if constexpr (details::is_map_v<T>) {
            return true;
        } else {
            return false;
        }
    }

    NODISCARD void* GetUnderlyingPtrImpl() const {
        return &value_;
    }

private:
    T value_;
    std::type_index type_index_;
};
#endif

class HolderBase {
public:
    virtual ~HolderBase() = default;
    virtual void CloneSmallObject(void*) const = 0;
    virtual void MoveSmallObject(void*) = 0;
    NODISCARD virtual std::unique_ptr<HolderBase> Clone() const = 0;
    NODISCARD virtual std::type_index type() const = 0;
    NODISCARD virtual uint32_t use_count() const = 0;
    NODISCARD virtual bool IsObjectRef() const = 0;
    NODISCARD virtual bool IsMap() const = 0;
    NODISCARD virtual void* GetUnderlyingPtr() = 0;
};

template<typename T>
class Holder final : public HolderBase {
public:
    explicit Holder(T value) : value_(std::move(value)) {}

    void CloneSmallObject(void* p) const override {
        new (p) Holder(*this);
    }

    void MoveSmallObject(void* p) override {
        new (p) Holder(std::move(*this));
    }

    NODISCARD std::unique_ptr<HolderBase> Clone() const override {
        return std::make_unique<Holder>(value_);
    }

    NODISCARD std::type_index type() const override {
        return typeid(T);
    }

    NODISCARD uint32_t use_count() const override {
        if constexpr (details::has_use_count_method_v<T>) {
            return value_.use_count();
        }
        return 1;
    }

    NODISCARD bool IsObjectRef() const override {
        if constexpr (std::is_base_of_v<ObjectRef, T>) {
            return true;
        }
        return false;
    }

    NODISCARD bool IsMap() const override {
        if constexpr (details::is_map_v<T>) {
            return true;
        }
        return false;
    }

    NODISCARD void* GetUnderlyingPtr() override {
        return &value_;
    }

private:
    T value_;

    friend class Any;
};

class Tensor;
class Function;
template<typename FType>
class TypedFunction;
template<typename>
class SingletonOrSharedTypePtr;
class Type;

class Any {
public:
    Any() = default;

#ifdef CPP20
    template<typename T, typename U = std::decay_t<T>>
        requires(!details::is_plain_type<U> && !std::same_as<U, Any>)
#else
    template<typename T,
             typename U = std::decay_t<T>,
             typename = std::enable_if_t<!details::is_plain_type<U> && !std::is_same_v<U, Any>>>
#endif
    Any(T&& value) : ptr_(std::make_unique<Holder<U>>(std::forward<T>(value))) {// NOLINT
    }

// integer ctor
#ifdef CPP20
    template<details::is_integral T>
#else
    template<typename T, std::enable_if_t<details::is_integral<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<int64_t>>(value)) {//NOLINT
    }

    // floating point ctor
#ifdef CPP20
    template<details::is_floating_point T>
#else
    template<typename T, std::enable_if_t<details::is_floating_point<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<double>>(value)) {//NOLINT
    }

    // string ctor
#ifdef CPP20
    template<details::is_string T>
#else
    template<typename T, std::enable_if_t<details::is_string<T>>* = nullptr>
#endif
    Any(T value) : ptr_(std::make_unique<Holder<String>>(std::move(value))) {//NOLINT
    }

    Any(const Any& other);

    Any(Any&& other) noexcept;

    Any& operator=(const Any& other) &;

    Any& operator=(Any&& other) & noexcept;

    template<typename T>
    Any& operator=(T value) & {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    NODISCARD void* GetUnderlyingPtr() const;

    /**
   * \brief Try to reinterpret the Any as a type T, return std::nullopt if it is not possible.
   *
   * \tparam T The type to cast to.
   * \return The cast value, or std::nullopt if the cast is not possible.
   * \note This function won't try to run type conversion (use try_cast for that purpose).
   */
#ifdef CPP20
    template<typename T>
        requires(!details::is_plain_type<T>)
#else
    template<typename T, std::enable_if_t<!details::is_plain_type<T>>* = nullptr>
#endif
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            if (has_value()) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return p->value_;
                }
            }
            return std::nullopt;
        }
    }

#ifdef CPP20
    template<typename T>
        requires(!details::is_plain_type<T>)
#else
    template<typename T, std::enable_if_t<!details::is_plain_type<T>>* = nullptr>
#endif
    std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return std::move(*this);
        } else {
            if (has_value()) {
                if (auto* p = dynamic_cast<Holder<T>*>(ptr_.get())) {
                    return std::move(p->value_);
                }
            }
            return std::nullopt;
        }
    }

#ifdef CPP20
    template<typename T>
        requires details::is_plain_type<T>
#else
    template<typename T, std::enable_if_t<details::is_plain_type<T>>* = nullptr>
#endif
    NODISCARD std::optional<T> as() const {
        if (has_value()) {
            if constexpr (details::is_integral<T>) {
                if (auto* p = dynamic_cast<Holder<int64_t>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_floating_point<T>) {
                if (auto* p = dynamic_cast<Holder<double>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            } else if constexpr (details::is_string<T>) {
                if (auto* p = dynamic_cast<Holder<String>*>(ptr_.get())) {
                    return static_cast<T>(p->value_);
                }
            }
        }
        return std::nullopt;
    }

    template<typename T>
    std::optional<T> try_cast() const {
        return as<T>();
    }

    template<typename T>
    NODISCARD T cast() const& {
        auto opt = as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError) << "cast failed.";
        }
        return opt.value();
    }

    template<typename T>
    T cast() && {
        auto opt = std::move(*this).as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError);
        }
        reset();
        return std::move(opt.value());
    }

    template<typename T>
    operator T() {//NOLINT
        return cast<T>();
    }


    template<typename T>
    T MoveFromAny() {
        return std::move(*this).cast<T>();
    }

    template<typename T>
        requires requires {
            typename T::size_type;
            requires details::is_array_subscript<T>;
        }
    decltype(auto) operator[](T::size_type i) {
        return (*static_cast<T*>(ptr_->GetUnderlyingPtr()))[i];
    }

    template<typename T>
        requires requires {
            typename T::key_type;
            typename T::mapped_type;
            requires details::is_map_subscript<T>;
        }
    decltype(auto) operator[](const T::key_type& key) {
        return (*static_cast<T*>(ptr_->GetUnderlyingPtr()))[key];
    }

    void reset();

    void swap(Any& other) noexcept;

    NODISCARD std::type_index type() const;

    NODISCARD SingletonOrSharedTypePtr<Type> GetTypePtr() const noexcept;

    NODISCARD bool has_value() const noexcept;

    NODISCARD bool IsNone() const noexcept;

    NODISCARD bool IsBool() const noexcept;

    NODISCARD bool IsInteger() const noexcept;

    NODISCARD bool IsFloatingPoint() const noexcept;

    NODISCARD bool IsString() const noexcept;

    NODISCARD bool IsVoidPtr() const noexcept;

    NODISCARD bool IsDevice() const noexcept;

    NODISCARD bool IsTensor() const noexcept;

    NODISCARD bool IsObjectRef() const noexcept;

    NODISCARD bool IsMap() const noexcept;

    NODISCARD String ToNone() const noexcept;

    NODISCARD int64_t ToInt() const;

    NODISCARD double ToDouble() const;

    NODISCARD bool ToBool() const;

    NODISCARD void* ToVoidPtr() const;

    NODISCARD Device ToDevice() const;

    NODISCARD String ToString() const;

    NODISCARD Tensor ToTensor() const;

    NODISCARD uint32_t use_count() const noexcept;

    NODISCARD bool unique() const noexcept;

    bool operator==(std::nullptr_t) const noexcept;

    bool operator!=(std::nullptr_t p) const noexcept;

private:
    std::unique_ptr<HolderBase> ptr_;

    friend class AnyEqual;
};

class AnyV1 {
public:
    using IntType = int64_t;
    using FloatType = double;
    using StringType = String;

    AnyV1() = default;
    ~AnyV1() = default;

    // not plain type ctor
    template<typename T, typename U = std::decay_t<T>>
        requires(!details::is_plain_type<U> && !std::same_as<U, AnyV1>)
    AnyV1(T&& value) {//NOLINT
        if constexpr (sizeof(U) <= kSmallObjectSize) {
            // small object, construct at local buffer
            data_ = SmallObject{std::forward<T>(value)};
        } else {
            data_ = std::make_unique<Holder<U>>(std::forward<T>(value));
        }
    }

    // integer ctor
    template<details::is_integral T>
    AnyV1(T value) {//NOLINT
        if constexpr (sizeof(Holder<IntType>) <= kSmallObjectSize) {
            // small object, construct at local buffer
            data_ = SmallObject{static_cast<IntType>(value)};
        } else {
            data_ = std::make_unique<Holder<IntType>>(static_cast<IntType>(value));
        }
    }

    // floating ctor
    template<details::is_floating_point T>
    AnyV1(T value) {//NOLINT
        if constexpr (sizeof(FloatType) <= kSmallObjectSize) {
            data_ = SmallObject{static_cast<FloatType>(value)};
        } else {
            data_ = std::make_unique<Holder<FloatType>>(static_cast<FloatType>(value));
        }
    }

    // string ctor
    template<details::is_string T>
    AnyV1(T value) {//NOLINT
        if constexpr (sizeof(StringType) <= kSmallObjectSize) {
            data_ = SmallObject{std::move(value)};
        } else {
            data_ = std::make_unique<Holder<StringType>>(static_cast<StringType>(value));
        }
    }

    AnyV1(const AnyV1& other) {
        if (other.IsSmallObject()) {
            data_ = std::get<SmallObject>(other.data_);
        } else if (other.IsLargeObject()) {
            data_ = std::get<std::unique_ptr<HolderBase>>(other.data_)->Clone();
        }
    }

    AnyV1(AnyV1&& other) noexcept : data_(std::move(other.data_)) {}

    AnyV1& operator=(const AnyV1& other) {
        AnyV1(other).swap(*this);
        return *this;
    }

    AnyV1& operator=(AnyV1&& other) noexcept {
        AnyV1(std::move(other)).swap(*this);
        return *this;
    }

    template<typename T>
    AnyV1& operator=(T value) & {
        AnyV1(std::move(value)).swap(*this);
        return *this;
    }

    void swap(AnyV1& other) noexcept {
        if (this == &other) {
            return;
        }
        std::swap(data_, other.data_);
    }

    void reset() noexcept {
        data_ = std::monostate{};
    }

    NODISCARD void* GetUnderlyingPtr() const {
        if (IsSmallObject()) {
            return std::get<SmallObject>(data_).GetUnderlyingPtr();
        }

        if (IsLargeObject()) {
            return std::get<std::unique_ptr<HolderBase>>(data_)->GetUnderlyingPtr();
        }

        return nullptr;
    }

    template<typename T>
        requires details::is_plain_type<T>
    NODISCARD std::optional<T> as() const {
        if (has_value()) {
            auto* p = GetUnderlyingPtr();
            if constexpr (details::is_integral<T>) {
                return static_cast<T>(*static_cast<IntType*>(p));
            } else if constexpr (details::is_floating_point<T>) {
                return static_cast<T>(*static_cast<FloatType*>(p));
            } else if constexpr (details::is_string<T>) {
                return static_cast<T>(*static_cast<StringType*>(p));
            }
        }

        return std::nullopt;
    }

    template<typename T>
        requires(!details::is_plain_type<T>)
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, AnyV1>) {
            return *this;
        } else {
            if (has_value()) {
                auto* p = GetUnderlyingPtr();
                return *static_cast<T*>(p);
            }
            return std::nullopt;
        }
    }

    template<typename T>
        requires(!details::is_plain_type<T>)
    NODISCARD std::optional<T> as() && {
        if constexpr (std::is_same_v<T, AnyV1>) {
            return std::move(*this);
        } else {
            if (has_value()) {
                auto* p = GetUnderlyingPtr();
                return std::move(*static_cast<T*>(p));
            }
            return std::nullopt;
        }
    }

    template<typename T>
    std::optional<T> try_cast() const {
        return as<T>();
    }

    template<typename T>
    NODISCARD T cast() const& {
        auto opt = as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError) << "cast failed.";
        }
        return opt.value();
    }

    template<typename T>
    T cast() && {
        auto opt = std::move(*this).as<T>();
        if (!opt.has_value()) {
            AETHERMIND_THROW(TypeError);
        }
        reset();
        return std::move(opt.value());
    }


    NODISCARD bool has_value() const noexcept {
        return !std::holds_alternative<std::monostate>(data_);
    }

    NODISCARD bool IsSmallObject() const {
        return std::holds_alternative<SmallObject>(data_);
    }

    NODISCARD bool IsLargeObject() const {
        return std::holds_alternative<std::unique_ptr<HolderBase>>(data_);
    }

    NODISCARD bool IsNone() const noexcept {
        return !has_value();
    }

private:
    // void InitLocalBuffer() noexcept {
    //     if (auto* p = std::get_if<SmallObject>(&data_)) {
    //         std::memset(p, 0, kSmallObjectSize);
    //     }
    // }

    static constexpr size_t kSmallObjectSize = sizeof(void*) * 2;

    struct SmallObject {
        SmallObject() : SmallObject(0) {}

        template<typename T, typename U = std::decay_t<T>>
            requires requires {
                requires !std::is_same_v<U, SmallObject>;
                requires sizeof(Holder<U>) <= kSmallObjectSize;
            }
        explicit SmallObject(T&& value) {
            new (local_buffer) Holder<U>(std::forward<T>(value));
        }

        SmallObject(const SmallObject& other) {
            const auto* src = reinterpret_cast<const HolderBase*>(other.local_buffer);
            src->CloneSmallObject(local_buffer);
        }

        SmallObject(SmallObject&& other) noexcept {
            auto* src = reinterpret_cast<HolderBase*>(other.local_buffer);
            src->MoveSmallObject(local_buffer);
        }

        SmallObject& operator=(const SmallObject& other) {
            SmallObject(other).swap(*this);
            return *this;
        }

        SmallObject& operator=(SmallObject&& other) noexcept {
            SmallObject(std::move(other)).swap(*this);
            return *this;
        }

        void swap(SmallObject& other) noexcept {
            using std::swap;
            uint8_t tmp[kSmallObjectSize];
            reinterpret_cast<HolderBase*>(local_buffer)->MoveSmallObject(tmp);
            reinterpret_cast<HolderBase*>(other.local_buffer)->MoveSmallObject(local_buffer);
            reinterpret_cast<HolderBase*>(tmp)->MoveSmallObject(other.local_buffer);
        }

        NODISCARD void* GetUnderlyingPtr() const {
            auto* p = reinterpret_cast<HolderBase*>(const_cast<uint8_t*>(local_buffer));
            return p->GetUnderlyingPtr();
        }

        ~SmallObject() {
            reinterpret_cast<HolderBase*>(local_buffer)->~HolderBase();
        }

        uint8_t local_buffer[kSmallObjectSize]{};
    };

    std::variant<std::monostate,
                 SmallObject,
                 std::unique_ptr<HolderBase>>
            data_{std::monostate{}};
};


// std::ostream& operator<<(std::ostream& os, const Any& any);

class AnyEqual {
public:
    bool operator()(const Any& lhs, const Any& rhs) const;
};

inline bool operator==(const Any& lhs, const Any& rhs) noexcept {
    return AnyEqual()(lhs, rhs);
}

inline bool operator!=(const Any& lhs, const Any& rhs) noexcept {
    return !AnyEqual()(lhs, rhs);
}

class AnyHash {
public:
    size_t operator()(const Any& v) const;
};

namespace details {

template<typename T>
struct TypeName {
    static String value() {
        return typeid(T).name();
    }
};

#define DEFINE_TYPE_NAME(code, bits, lanes, T, name) \
    template<>                                       \
    struct TypeName<T> {                             \
        static String value() {                      \
            return #name;                            \
        }                                            \
    };

SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(DEFINE_TYPE_NAME);
DEFINE_TYPE_NAME(_, _, _, Tensor, Tensor);
DEFINE_TYPE_NAME(_, _, _, Device, Device);
DEFINE_TYPE_NAME(_, _, _, Any, Any);
DEFINE_TYPE_NAME(_, _, _, Any*, Any*);
DEFINE_TYPE_NAME(_, _, _, const Any*, const Any*);
DEFINE_TYPE_NAME(_, _, _, const Any&, const Any&);
DEFINE_TYPE_NAME(_, _, _, void, void);
DEFINE_TYPE_NAME(_, _, _, Function, Function);

template<typename FType>
struct TypeName<TypedFunction<FType>> {
    static String value() {
        return "Function";
    }
};

#undef DEFINE_TYPE_NAME

template<typename T>
struct Type2Str {
    using U = std::remove_const_t<std::remove_reference_t<T>>;
    static String value() {
        return TypeName<U>::value();
    }
};

}// namespace details

// template<typename T>
//     requires std::default_initializable<T>
// class Test {
// public:
//     Test() = default;
//
//     void print();
//
// private:
//     T value;
// };
//
// template<typename T>
//     requires std::default_initializable<T>
// void Test<T>::print() {
//     std::cout << value;
// }

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::Any> {
    size_t operator()(const aethermind::Any& v) const noexcept {
        return aethermind::AnyHash()(v);
    }
};
}// namespace std

#endif//AETHERMIND_ANY_H
