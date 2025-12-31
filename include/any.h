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
    NODISCARD virtual void* GetDataPtr() = 0;
    NODISCARD virtual const void* GetDataPtr() const = 0;
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
        } else {
            return 1;
        }
    }

    NODISCARD bool IsObjectRef() const override {
        if constexpr (std::is_base_of_v<ObjectRef, T>) {
            return true;
        } else {
            return false;
        }
    }

    NODISCARD bool IsMap() const override {
        if constexpr (details::is_map_v<T>) {
            return true;
        } else {
            return false;
        }
    }

    NODISCARD void* GetDataPtr() override {
        return &value_;
    }

    NODISCARD const void* GetDataPtr() const override {
        return &value_;
    }

private:
    T value_;
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
    using Int = int64_t;
    using Float = double;
    using Bool = bool;

    Any() = default;
    ~Any() = default;

    // not plain type ctor
    template<typename T, typename U = std::decay_t<T>>
        requires(!details::is_plain_type<U> && !std::same_as<U, Any>)
    Any(T&& value) {//NOLINT
        if constexpr (sizeof(Holder<U>) <= kSmallObjectSize) {
            // small object, construct at local buffer
            data_ = SmallObject{std::forward<T>(value)};
        } else {
            data_ = std::make_unique<Holder<U>>(std::forward<T>(value));
        }
    }

    // integer ctor
    template<details::is_integral T>
    Any(T value) {//NOLINT
        if constexpr (sizeof(Holder<Int>) <= kSmallObjectSize) {
            // small object, construct at local buffer
            data_ = SmallObject{static_cast<Int>(value)};
        } else {
            data_ = std::make_unique<Holder<Int>>(static_cast<Int>(value));
        }
    }

    // boolean ctor
    template<details::is_boolean T>
    Any(T value) {//NOLINT
        if constexpr (sizeof(Holder<Bool>) <= kSmallObjectSize) {
            // small object, construct at local buffer
            data_ = SmallObject{static_cast<Bool>(value)};
        } else {
            data_ = std::make_unique<Holder<Bool>>(static_cast<Bool>(value));
        }
    }

    // floating ctor
    template<details::is_floating_point T>
    Any(T value) {//NOLINT
        if constexpr (sizeof(Holder<Float>) <= kSmallObjectSize) {
            data_ = SmallObject{static_cast<Float>(value)};
        } else {
            data_ = std::make_unique<Holder<Float>>(static_cast<Float>(value));
        }
    }

    // string ctor
    template<details::is_string T>
    Any(T value) {//NOLINT
        if constexpr (sizeof(Holder<String>) <= kSmallObjectSize) {
            data_ = SmallObject{std::move(value)};
        } else {
            data_ = std::make_unique<Holder<String>>(static_cast<String>(value));
        }
    }

    Any(const Any& other);

    Any(Any&& other) noexcept : data_(std::move(other.data_)) {}

    Any& operator=(const Any& other) {
        Any(other).swap(*this);
        return *this;
    }

    Any& operator=(Any&& other) noexcept {
        Any(std::move(other)).swap(*this);
        return *this;
    }

    template<typename T>
    Any& operator=(T value) & {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    void swap(Any& other) noexcept {
        if (this == &other) {
            return;
        }
        std::swap(data_, other.data_);
    }

    void reset() noexcept {
        data_ = std::monostate{};
    }

    NODISCARD const HolderBase* GetHolderPtr() const;

    NODISCARD void* GetDataPtr();

    NODISCARD const void* GetDataPtr() const;

    template<details::is_plain_type T>
    NODISCARD std::optional<T> as() const {
        // auto visitor = []<typename Arg>(const Arg& arg) -> std::optional<T> {
        //     using U = std::decay_t<Arg>;
        //     if constexpr (std::is_same_v<U, SmallObject>) {
        //         //
        //     } else if constexpr (std::is_same_v<U, std::unique_ptr<HolderBase>>) {
        //         //
        //     } else {
        //         return std::nullopt;
        //     }
        // };

        if (has_value()) {
            const auto* p = GetDataPtr();
            if constexpr (details::is_integral<T>) {
                if (IsInteger()) {
                    return static_cast<T>(*static_cast<const Int*>(p));
                }
            } else if constexpr (details::is_boolean<T>) {
                if (IsBool()) {
                    return static_cast<T>(*static_cast<const Bool*>(p));
                }
            } else if constexpr (details::is_floating_point<T>) {
                if (IsFloatingPoint()) {
                    return static_cast<T>(*static_cast<const Float*>(p));
                }
            } else if constexpr (details::is_string<T>) {
                if (IsString()) {
                    return static_cast<T>(*static_cast<const String*>(p));
                }
            }
        }

        return std::nullopt;
    }

    template<typename T>
        requires(!details::is_plain_type<T>)
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            if (has_value()) {
                if (dynamic_cast<const Holder<T>*>(GetHolderPtr())) {
                    return *static_cast<const T*>(GetDataPtr());
                }
            }
            return std::nullopt;
        }
    }

    template<typename T>
        requires(!details::is_plain_type<T>)
    NODISCARD std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return std::move(*this);
        } else {
            if (has_value()) {
                if (dynamic_cast<const Holder<T>*>(GetHolderPtr())) {
                    return std::move(*static_cast<T*>(GetDataPtr()));
                }
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
        return (*static_cast<T*>(GetDataPtr()))[i];
    }

    template<typename T>
        requires requires {
            typename T::key_type;
            typename T::mapped_type;
            requires details::is_map_subscript<T>;
        }
    decltype(auto) operator[](const T::key_type& key) {
        return (*static_cast<T*>(GetDataPtr()))[key];
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

    NODISCARD std::type_index type() const;

    NODISCARD SingletonOrSharedTypePtr<Type> GetTypePtr() const noexcept;

    NODISCARD bool IsNone() const noexcept {
        return !has_value();
    }

    NODISCARD bool IsBool() const noexcept {
        return type() == std::type_index(typeid(Bool));
    }

    NODISCARD bool IsInteger() const noexcept {
        return type() == std::type_index(typeid(Int));
    }

    NODISCARD bool IsFloatingPoint() const noexcept {
        return type() == std::type_index(typeid(Float));
    }

    NODISCARD bool IsString() const noexcept {
        return type() == std::type_index(typeid(String));
    }

    NODISCARD bool IsVoidPtr() const noexcept {
        return type() == std::type_index(typeid(void*));
    }

    NODISCARD bool IsDevice() const noexcept {
        return type() == std::type_index(typeid(Device));
    }

    NODISCARD bool IsTensor() const noexcept;

    NODISCARD bool IsObjectRef() const noexcept {
        return has_value() ? GetHolderPtr()->IsObjectRef() : false;
    }

    NODISCARD bool IsMap() const noexcept {
        return has_value() ? GetHolderPtr()->IsMap() : false;
    }

    NODISCARD String ToNone() const noexcept {
        CHECK(IsNone());
        return "None";
    }

    NODISCARD Int ToInt() const {
        CHECK(IsInteger()) << "Expected Int.";
        return cast<Int>();
    }

    NODISCARD Float ToDouble() const {
        CHECK(IsFloatingPoint()) << "Expected Double.";
        return cast<double>();
    }

    NODISCARD bool ToBool() const {
        CHECK(IsBool()) << "Expected Bool.";
        return cast<bool>();
    }

    NODISCARD void* ToVoidPtr() const {
        CHECK(IsVoidPtr()) << "Expected VoidPtr.";
        return cast<void*>();
    }

    NODISCARD Device ToDevice() const {
        CHECK(IsDevice()) << "Expected Device.";
        return cast<Device>();
    }

    NODISCARD String ToString() const {
        CHECK(IsString()) << "Expected String.";
        return cast<String>();
    }

    NODISCARD Tensor ToTensor() const;

    NODISCARD uint32_t use_count() const noexcept {
        return has_value() ? GetHolderPtr()->use_count() : 0;
    }

    NODISCARD bool unique() const noexcept {
        return use_count() == 1;
    }

    bool operator==(std::nullptr_t) const noexcept {
        return has_value() ? type() == std::type_index(typeid(std::nullptr_t)) : true;
    }

    bool operator!=(std::nullptr_t p) const noexcept {
        return !(*this == p);
    }

private:
    static constexpr size_t kSmallObjectSize = sizeof(void*) * 2;

    // template<typename R, typename F>
    // std::optional<R> visitor(F&& func) const {
    //
    // }

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

        NODISCARD void* GetDataPtr() {
            return reinterpret_cast<HolderBase*>(local_buffer)->GetDataPtr();
        }

        NODISCARD const void* GetDataPtr() const {
            return reinterpret_cast<const HolderBase*>(local_buffer)->GetDataPtr();
        }

        ~SmallObject() {
            reinterpret_cast<HolderBase*>(local_buffer)->~HolderBase();
        }

        uint8_t local_buffer[kSmallObjectSize]{};
    };

    template<typename T>
    struct Caster {
        std::optional<T> operator()(const std::monostate&) const {
            return std::nullopt;
        }

        std::optional<T> operator()(const SmallObject& v) const {
            // if constexpr (details::is_integral<T>) {
            //     return
            // }
        }

        std::optional<T> operator()(const std::unique_ptr<HolderBase>& v) const {
            //
        }
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
