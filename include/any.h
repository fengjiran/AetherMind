//
// Created by 赵丹 on 2025/8/15.
//

#ifndef AETHERMIND_ANY_H
#define AETHERMIND_ANY_H

#include "any_utils.h"
#include "object.h"

#include <optional>
#include <typeindex>
#include <variant>

// #define USE_VISITOR_PATTERN

namespace aethermind {

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
    NODISCARD virtual bool EqualsTo(const HolderBase*) const = 0;
    virtual void print(std::ostream& os) const = 0;
};

template<typename T>
class Holder final : public HolderBase {
public:
    explicit Holder(T value) : value_(std::move(value)) {}
    // Holder(T&& value) : value_(std::forward<T>(value)) {}

    operator T&() noexcept {
        return value_;
    }

    operator const T&() const noexcept {
        return value_;
    }

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
        if constexpr (details::is_map<T>) {
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

    NODISCARD bool EqualsTo(const HolderBase* other) const override {
        if constexpr (std::equality_comparable<T>) {
            return value_ == *static_cast<const T*>(other->GetDataPtr());
        } else {
            return GetDataPtr() == other->GetDataPtr();
        }
    }

    void print(std::ostream& os) const override {
        if constexpr (details::is_printable<T>) {
            os << value_;
        } else {
            // print type info and address
            os << "[" << details::demangle(type().name()) << "@" << std::hex
               << reinterpret_cast<std::uintptr_t>(GetDataPtr()) << "]";
        }
    }

private:
    T value_;
};

template<typename>
class SingletonOrSharedTypePtr;
class Type;

class Any {
public:
    using Int = int64_t;
    using Float = double;
    using Bool = bool;
    template<typename T>
    using select_plain_type = std::conditional_t<details::is_boolean<T>, Bool,
                                                 std::conditional_t<details::is_integral<T>,
                                                                    Int, std::conditional_t<details::is_floating_point<T>, Float, String>>>;
    Any() = default;
    ~Any() = default;

    template<typename T,
             typename U = std::decay_t<T>,
             typename TargetType = std::conditional_t<details::is_plain_type<U>, select_plain_type<U>, U>>
        requires(!std::is_same_v<U, Any>)
    Any(T&& value) : type_info_cache_(std::type_index(typeid(TargetType))) {//NOLINT
        if constexpr (details::is_plain_type<U>) {
            if constexpr (sizeof(Holder<TargetType>) <= kSmallObjectSize) {
                // small object, construct at local buffer
                data_ = SmallObject{static_cast<TargetType>(value)};
            } else {
                data_ = std::make_unique<Holder<TargetType>>(static_cast<TargetType>(value));
            }
        } else {
            if constexpr (sizeof(Holder<TargetType>) <= kSmallObjectSize) {
                // small object, construct at local buffer
                data_ = SmallObject{std::forward<T>(value)};
            } else {
                data_ = std::make_unique<Holder<TargetType>>(std::forward<T>(value));
            }
        }
    }

    Any(const Any& other);

    Any(Any&& other) noexcept
        : data_(std::move(other.data_)), type_info_cache_(other.type_info_cache_) {
        other.type_info_cache_ = std::type_index(typeid(void));
    }

    Any& operator=(const Any& other) {
        Any(other).swap(*this);
        return *this;
    }

    Any& operator=(Any&& other) noexcept {
        Any(std::move(other)).swap(*this);
        return *this;
    }

    template<typename T>
        requires requires(T t) {
            Any(t);
        }
    Any& operator=(T value) & {
        Any(std::move(value)).swap(*this);
        return *this;
    }

    void swap(Any& other) noexcept {
        if (this == &other) {
            return;
        }
        std::swap(data_, other.data_);
        std::swap(type_info_cache_, other.type_info_cache_);
    }

    void reset() noexcept {
        data_ = std::monostate{};
        type_info_cache_ = std::type_index(typeid(void));
    }

    NODISCARD const HolderBase* GetHolderPtr() const;

    NODISCARD void* GetDataPtr();

    NODISCARD const void* GetDataPtr() const;

#ifdef USE_VISITOR_PATTERN
    template<typename T>
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            return std::visit(Caster<T>{}, data_);
        }
    }

    template<typename T>
    NODISCARD std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return std::move(*this);
        } else {
            return std::visit(Caster<T>{}, std::move(data_));
        }
    }
#else
    template<typename T>
    NODISCARD std::optional<T> as() const& {
        if constexpr (std::is_same_v<T, Any>) {
            return *this;
        } else {
            switch (data_.index()) {
                case 0:// std::monostate
                    return std::nullopt;
                case 1:// small object
                    return Caster<T>()(std::get<SmallObject>(data_));
                case 2:// large object
                    return Caster<T>()(std::get<std::unique_ptr<HolderBase>>(data_));
                default:
                    return std::nullopt;
            }
        }
    }

    template<typename T>
    NODISCARD std::optional<T> as() && {
        if constexpr (std::is_same_v<T, Any>) {
            return std::move(*this);
        } else {
            switch (data_.index()) {
                case 0:// std::monostate
                    return std::nullopt;
                case 1:// small object
                    return Caster<T>()(std::move(std::get<SmallObject>(data_)));
                case 2:// large object
                    return Caster<T>()(std::move(std::get<std::unique_ptr<HolderBase>>(data_)));
                default:
                    return std::nullopt;
            }
        }
    }

#endif

    template<typename T>
    NODISCARD bool can_cast() const noexcept {
        return as<T>().has_value();
    }

    template<typename T>
    NODISCARD std::optional<T> try_cast() const {
        return as<T>();
    }

    template<typename T>
    NODISCARD T cast() const& {
        auto opt = as<T>();
        if (opt.has_value()) {
            return *opt;
        }
        AETHERMIND_THROW(TypeError) << "Cast failed to type: " << details::Type2Str<T>::value();
        AETHERMIND_UNREACHABLE();
    }

    template<typename T>
    T cast() && {
        if (auto opt = std::move(*this).as<T>(); opt.has_value()) {
            reset();
            return std::move(opt.value());
        }
        AETHERMIND_THROW(TypeError) << "Cast failed to type: " << details::Type2Str<T>::value();
        AETHERMIND_UNREACHABLE();
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
        requires details::is_container<T>
    decltype(auto) operator[](T::size_type i) {
        CHECK(has_value()) << "Any has no value.";
        CHECK(CheckType<T>()) << "Type mismatch.";
        return (*static_cast<T*>(GetDataPtr()))[i];
    }

    template<typename T>
        requires details::is_map<T>
    decltype(auto) operator[](const T::key_type& key) {
        return (*static_cast<T*>(GetDataPtr()))[key];
    }

    NODISCARD bool has_value() const noexcept {
        return !std::holds_alternative<std::monostate>(data_);
    }

    NODISCARD bool IsSmallObject() const noexcept {
        return std::holds_alternative<SmallObject>(data_);
        // return data_.index() == 1;
    }

    NODISCARD bool IsLargeObject() const noexcept {
        return std::holds_alternative<std::unique_ptr<HolderBase>>(data_);
        // return data_.index() == 2;
    }

    NODISCARD std::type_index type() const;

    NODISCARD SingletonOrSharedTypePtr<Type> GetTypePtr() const noexcept;

    NODISCARD bool IsNone() const noexcept {
        return !has_value();
    }

    NODISCARD bool IsBool() const noexcept {
        return CheckType<Bool>();
    }

    NODISCARD bool IsInteger() const noexcept {
        return CheckType<Int>();
    }

    NODISCARD bool IsFloatingPoint() const noexcept {
        return CheckType<Float>();
    }

    NODISCARD bool IsString() const noexcept {
        return CheckType<String>();
    }

    NODISCARD bool IsVoidPtr() const noexcept {
        return CheckType<void*>();
    }

    NODISCARD bool IsDevice() const noexcept {
        return CheckType<Device>();
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
        CHECK(IsInteger()) << "Expected Int, but got " << type().name();
        return cast<Int>();
    }

    NODISCARD Float ToDouble() const {
        CHECK(IsFloatingPoint()) << "Expected Double, but got " << type().name();
        return cast<Float>();
    }

    NODISCARD Bool ToBool() const {
        CHECK(IsBool()) << "Expected Bool, but got " << type().name();
        return cast<Bool>();
    }

    NODISCARD void* ToVoidPtr() const {
        CHECK(IsVoidPtr()) << "Expected VoidPtr, but got " << type().name();
        return cast<void*>();
    }

    NODISCARD Device ToDevice() const {
        CHECK(IsDevice()) << "Expected Device, but got " << type().name();
        return cast<Device>();
    }

    NODISCARD String ToString() const {
        CHECK(IsString()) << "Expected String, but got " << type().name();
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

    void DebugPrint(std::ostream& os) const;

private:
    static constexpr size_t kSmallObjectSize = sizeof(void*) * 2;

    template<typename T>
    NODISCARD bool CheckType() const noexcept {
        return type() == std::type_index(typeid(T));
    }

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
            other.GetHolderPtr()->CloneSmallObject(local_buffer);
        }

        SmallObject(SmallObject&& other) noexcept {
            other.GetHolderPtr()->MoveSmallObject(local_buffer);
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
            if (this == &other) {
                return;
            }

            uint8_t tmp[kSmallObjectSize];
            std::memcpy(tmp, local_buffer, kSmallObjectSize);
            std::memcpy(local_buffer, other.local_buffer, kSmallObjectSize);
            std::memcpy(other.local_buffer, tmp, kSmallObjectSize);
        }

        NODISCARD HolderBase* GetHolderPtr() {
            return reinterpret_cast<HolderBase*>(local_buffer);
        }

        NODISCARD const HolderBase* GetHolderPtr() const {
            return reinterpret_cast<const HolderBase*>(local_buffer);
        }

        NODISCARD void* GetDataPtr() {
            return GetHolderPtr()->GetDataPtr();
        }

        NODISCARD const void* GetDataPtr() const {
            return GetHolderPtr()->GetDataPtr();
        }

        ~SmallObject() {
            GetHolderPtr()->~HolderBase();
        }

        // alignas(std::max_align_t)
        uint8_t local_buffer[kSmallObjectSize]{};
    };

    template<typename T>
    class Caster {
    public:
        template<typename U>
        std::optional<T> operator()(U&& v) const {
            using V = std::decay_t<U>;
            if constexpr (std::is_same_v<V, std::monostate>) {
                return std::nullopt;
            } else if constexpr (std::is_same_v<V, SmallObject>) {
                return CastImpl(v.GetHolderPtr());
            } else if constexpr (std::is_same_v<V, std::unique_ptr<HolderBase>>) {
                if constexpr (std::is_const_v<std::remove_reference_t<U>>) {
                    const auto* holder_ptr = v.get();
                    return CastImpl(holder_ptr);
                } else {
                    auto* holder_ptr = v.get();
                    return CastImpl(holder_ptr);
                }
            }
            return std::nullopt;
        }

    private:
        template<typename HolderType>
        static std::optional<T> CastImpl(HolderType* holder_ptr) {
            if constexpr (details::is_integral<T>) {
                if (holder_ptr->type() == std::type_index(typeid(Int))) {
                    return static_cast<T>(*static_cast<const Int*>(holder_ptr->GetDataPtr()));
                }
            } else if constexpr (details::is_boolean<T>) {
                if (holder_ptr->type() == std::type_index(typeid(Bool))) {
                    return static_cast<T>(*static_cast<const Bool*>(holder_ptr->GetDataPtr()));
                }
            } else if constexpr (details::is_floating_point<T>) {
                if (holder_ptr->type() == std::type_index(typeid(Float))) {
                    return static_cast<T>(*static_cast<const Float*>(holder_ptr->GetDataPtr()));
                }
            } else if constexpr (details::is_string<T>) {
                if (holder_ptr->type() == std::type_index(typeid(String))) {
                    if constexpr (std::is_same_v<HolderType, const HolderBase>) {
                        return static_cast<T>(*static_cast<const String*>(holder_ptr->GetDataPtr()));
                    } else {
                        return static_cast<T>(std::move(*static_cast<String*>(holder_ptr->GetDataPtr())));
                    }
                }
            } else {
                if (holder_ptr->type() == std::type_index(typeid(T))) {
                    if constexpr (std::is_same_v<HolderType, const HolderBase>) {
                        return static_cast<T>(*static_cast<const T*>(holder_ptr->GetDataPtr()));
                    } else {
                        if constexpr (std::is_move_constructible_v<T>) {
                            return static_cast<T>(std::move(*static_cast<T*>(holder_ptr->GetDataPtr())));
                        } else {
                            return static_cast<T>(*static_cast<T*>(holder_ptr->GetDataPtr()));
                        }
                    }
                }
            }

            return std::nullopt;
        }
    };

    std::variant<std::monostate,
                 SmallObject,
                 std::unique_ptr<HolderBase>>
            data_{std::monostate{}};
    mutable std::type_index type_info_cache_{std::type_index(typeid(void))};
};

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

enum class AnyPrintFormat {
    Default,// default format
    Debug,  // debug format, include type info
    Compact // compact format, only value
};

std::ostream& operator<<(std::ostream& os, const Any& any);

inline void PrintAny(std::ostream& os, const Any& any, AnyPrintFormat format = AnyPrintFormat::Default) {
    switch (format) {
        case AnyPrintFormat::Debug:
            any.DebugPrint(os);
            break;
        case AnyPrintFormat::Compact:
        default:
            os << any;
            break;
    }
}

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
