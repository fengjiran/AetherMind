//
// Created by richard on 9/24/25.
//

#ifndef AETHERMIND_FUNCTION_H
#define AETHERMIND_FUNCTION_H

#include "any.h"
#include "function_traits.h"
#include "object.h"

#include <functional>
#include <string>
#include <vector>

namespace aethermind {

namespace details {

template<typename F, typename... Args>
void for_each(const F& f, Args&&... args) {
    int i = 0;
    (f(i++, std::forward<Args>(args)), ...);
}

class PackedArgsSetter {
public:
    explicit PackedArgsSetter(Any* args) : args_(args) {}

    template<typename T>
    void operator()(size_t i, T&& value) const {
        args_[i].operator=(std::forward<T>(value));
    }

private:
    Any* args_;
};

class PackedArgs {
public:
    PackedArgs(const Any* args, int32_t size) : args_(args), size_(size) {}

    int32_t size() const {
        return size_;
    }

    const Any* data() const {
        return args_;
    }

    Any operator[](size_t i) const {
        return args_[i];
    }

    PackedArgs slice(int begin, int end = -1) const {
        if (end == -1) {
            end = size_;
        }
        return {args_ + begin, end - begin};
    }

    template<typename... Args>
    static void Fill(Any* data, Args&&... args) {
        for_each(PackedArgsSetter(data), std::forward<Args>(args)...);
    }

private:
    const Any* args_;
    int32_t size_;
};
}// namespace details

class FunctionImpl : public Object {
public:
    using FCall = std::function<void(const FunctionImpl*, const Any*, int32_t, Any*)>;

    FunctionImpl() : callable_(nullptr) {}

    explicit FunctionImpl(FCall callable) : callable_(std::move(callable)) {}

    void CallPacked(const Any* args, int32_t num_args, Any* res) const {
        callable_(this, args, num_args, res);
    }

    template<typename TCallable>
    static ObjectPtr<FunctionImpl> create(TCallable packed_call) {
        static_assert(std::is_convertible_v<TCallable, std::function<void(const Any*, int32_t, Any*)>> ||
                              std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>,
                      "FromPacked requires input function signature to match packed func format");
        if constexpr (std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>) {
            auto f = [packed_call](const FunctionImpl*, const Any* args, int32_t num_args, Any* res) -> void {
                details::PackedArgs packed_args(args, num_args);
                packed_call(packed_args, res);
            };
            return make_object<FunctionImpl>(f);
        } else {
            auto f = [packed_call](const FunctionImpl*, const Any* args, int32_t num_args, Any* res) -> void {
                packed_call(args, num_args, res);
            };
            return make_object<FunctionImpl>(f);
        }
    }

private:
    FCall callable_;
};


class Function {
public:
    Function() = default;

    explicit Function(ObjectPtr<FunctionImpl> impl) : pimpl_(std::move(impl)) {}

    template<typename TCallable>
    explicit Function(TCallable packed_call) : pimpl_(FunctionImpl::create(packed_call)) {}

    template<typename TCallable>
    static Function FromPacked(TCallable packed_call) {
        return Function(packed_call);
    }

    NODISCARD bool defined() const noexcept;

    NODISCARD uint32_t use_count() const noexcept;

    NODISCARD bool unique() const noexcept;

    NODISCARD FunctionImpl* get_impl_ptr_unsafe() const noexcept;

    NODISCARD FunctionImpl* release_impl_unsafe() noexcept;

    template<typename... Args>
    Any operator()(Args&&... args) const {
        const int num_args = sizeof...(Args);
        const int array_size = num_args > 0 ? num_args : 1;
        Any args_pack[array_size];
        details::PackedArgs::Fill(args_pack, std::forward<Args>(args)...);
        Any res;
        pimpl_->CallPacked(args_pack, num_args, &res);
        return res;
    }

    void CallPacked(const Any* args, int32_t num_args, Any* res) const;

    void CallPacked(details::PackedArgs args, Any* res) const;

private:
    ObjectPtr<FunctionImpl> pimpl_;
};

template<typename FType>
class TypedFunction;

template<typename R, typename... Args>
class TypedFunction<R(Args...)> {
public:
    using Self = TypedFunction;

    TypedFunction() = default;

    TypedFunction(std::nullopt_t) {} // NOLINT

    TypedFunction(Function packed_func) : packed_func_(std::move(packed_func)) {} //NOLINT

private:
    Function packed_func_;
};

template<>
struct TypeTraits<Function> : TypeTraitsBase {
    static void CopyToAny(const Function& src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Function;
        FunctionImpl* obj = src.get_impl_ptr_unsafe();
        dst->payload_ = obj;
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRef(obj);
        }
    }

    static void MoveToAny(Function src, AetherMindAny* dst) {
        dst->tag_ = AnyTag::Function;
        dst->payload_ = static_cast<Object*>(src.release_impl_unsafe());
    }

    static Function CopyFromAnyAfterCheck(const AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        if (!IsNullTypePtr(obj)) {
            details::ObjectUnsafe::IncRef(obj);
        }
        return Function(ObjectPtr<FunctionImpl>::reclaim(static_cast<FunctionImpl*>(obj)));
    }

    static Function MoveFromAnyAfterCheck(AetherMindAny* src) {
        auto* obj = std::get<Object*>(src->payload_);
        src->payload_ = 0;
        src->tag_ = AnyTag::None;
        return Function(ObjectPtr<FunctionImpl>::reclaim(static_cast<FunctionImpl*>(obj)));
    }

    static std::optional<Function> TryCastFromAny(const AetherMindAny* src) {
        if (check(src)) {
            return CopyFromAnyAfterCheck(src);
        }
        return std::nullopt;
    }

    static bool check(const AetherMindAny* src) {
        return src->tag_ == AnyTag::Function;
    }

    static std::string TypeStr() {
        return AnyTagToString(AnyTag::Function);
    }
};

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_H
