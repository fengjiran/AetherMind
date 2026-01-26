//
// Created by richard on 9/24/25.
//

#ifndef AETHERMIND_FUNCTION_H
#define AETHERMIND_FUNCTION_H

#include "container/array.h"
#include "function_traits.h"
#include "utils/qualified_name.h"

#include <functional>
#include <utility>

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

    AM_NODISCARD int32_t size() const {
        return size_;
    }

    AM_NODISCARD const Any* data() const {
        return args_;
    }

    Any operator[](size_t i) const {
        return args_[i];
    }

    AM_NODISCARD PackedArgs slice(int begin, int end = -1) const {
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

template<typename R, size_t... Is, typename F>
void unpack_call(const F& callable, std::index_sequence<Is...>, const String* opt_name,
                 const Any* args, int32_t num_args, Any* res) {
    using FuncInfo = FunctionInfo<F>;
    const FGetFunctionSchema f_schema = FuncInfo::Schema;
    // static_assert(FuncInfo::unpacked_args_supported, "the function signature do not support unpacked.");
    constexpr size_t nargs = sizeof...(Is);
    if (nargs != num_args) {
        AM_THROW(TypeError) << "Mismatched number of arguments when calling: `"
                                    << (opt_name == nullptr ? "" : *opt_name)
                                    << (f_schema == nullptr ? "" : (*f_schema)()) << "`. Expected " << nargs
                                    << " but got " << num_args << " arguments";
    }

    if constexpr (std::is_same_v<R, void>) {
        callable(Any2Arg(args, Is, opt_name, f_schema)...);
    } else {
        *res = R(callable(Any2Arg(args, Is, opt_name, f_schema)...));
    }
}

}// namespace details

class FunctionImpl : public Object {
public:
    using FCall = std::function<void(const FunctionImpl*, const Any*, int32_t, Any*)>;

    FunctionImpl() : callable_(nullptr), qualified_name_("no_qualified_name"), schema_("") {}

    explicit FunctionImpl(FCall callable, QualifiedName qualified_name = "no_qualified_name", String schema = "")
        : callable_(std::move(callable)), qualified_name_(std::move(qualified_name)), schema_(std::move(schema)) {}

    void CallPacked(const Any* args, int32_t num_args, Any* res) const {
        callable_(this, args, num_args, res);
    }

    AM_NODISCARD const String& schema() const {
        return schema_;
    }

    AM_NODISCARD const QualifiedName& GetQualifiedName() const {
        return qualified_name_;
    }

    AM_NODISCARD const String& GetName() const {
        return qualified_name_.GetName();
    }

    template<typename TCallable>
    static ObjectPtr<FunctionImpl> Create(TCallable packed_call,
                                          const QualifiedName& qualified_name = "no_qualified_name",
                                          const String& schema = "") {
        static_assert(std::is_convertible_v<TCallable, std::function<void(const Any*, int32_t, Any*)>> ||
                              std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>,
                      "FromPacked requires input function signature to match packed func format");
        if constexpr (std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>) {
            auto f = [packed_call](const FunctionImpl*, const Any* args, int32_t num_args, Any* res) mutable -> void {
                details::PackedArgs packed_args(args, num_args);
                packed_call(packed_args, res);
            };
            return make_object<FunctionImpl>(f, qualified_name, schema);
        } else {
            auto f = [packed_call](const FunctionImpl*, const Any* args, int32_t num_args, Any* res) mutable -> void {
                packed_call(args, num_args, res);
            };
            return make_object<FunctionImpl>(f, qualified_name, schema);
        }
    }

private:
    FCall callable_;
    QualifiedName qualified_name_;
    String schema_;
};

class Function : public ObjectRef {
public:
    Function() = default;

    explicit Function(ObjectPtr<FunctionImpl> impl) : pimpl_(std::move(impl)) {}

    template<typename TCallable>
    Function(TCallable callable, const QualifiedName& qualified_name = "no_qualified_name") {
        if constexpr (std::is_convertible_v<TCallable, std::function<void(const Any*, int32_t, Any*)>> ||
                      std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>) {
            *this = FromPacked(callable, qualified_name);
        } else {
            *this = FromTyped(callable, qualified_name);
        }
    }

    Function(const Function&) = default;
    Function(Function&&) noexcept = default;
    Function& operator=(const Function&) = default;
    Function& operator=(Function&&) noexcept = default;

    template<typename TCallable>
    static Function FromPacked(TCallable packed_call, const QualifiedName& qualified_name) {
        String schema;
        if constexpr (std::is_convertible_v<TCallable, std::function<void(details::PackedArgs args, Any*)>>) {
            schema = "(0: PackedArgs, 1: Any*) -> void";
        } else if constexpr (std::is_convertible_v<TCallable, std::function<void(const Any*, int32_t, Any*)>>) {
            schema = "(0: const Any*, 1: Int, 2: Any*) -> void";
        }
        return Function(packed_call, qualified_name, schema);
    }

    template<typename TCallable>
    static Function FromTyped(TCallable callable, const QualifiedName& qualified_name) {
        using FuncInfo = details::FunctionInfo<TCallable>;
        using R = FuncInfo::return_type;
        using idx_seq = std::make_index_sequence<FuncInfo::num_args>;
        auto f = [callable, qualified_name](const Any* args, int32_t num_args, Any* res) mutable -> void {
            details::unpack_call<R>(callable, idx_seq{}, &qualified_name.GetQualifiedName(), args, num_args, res);
        };
        return Function(f, qualified_name, FuncInfo::Schema());
    }

    AM_NODISCARD static std::optional<Function> GetGlobalFunction(const String& name);

    AM_NODISCARD static Function GetGlobalFunctionRequired(const String& name);

    AM_NODISCARD static Array<String> ListGlobalFunctionNames();

    AM_NODISCARD bool defined() const noexcept;

    AM_NODISCARD uint32_t use_count() const noexcept;

    AM_NODISCARD bool unique() const noexcept;

    AM_NODISCARD const String& schema() const noexcept;

    AM_NODISCARD const QualifiedName& GetQualifiedName() const noexcept;

    AM_NODISCARD FunctionImpl* get_impl_ptr_unsafe() const noexcept;

    AM_NODISCARD FunctionImpl* release_impl_unsafe() noexcept;

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
    template<typename TCallable>
    Function(TCallable packed_call, const QualifiedName& qualified_name, const String& schema)
        : pimpl_(FunctionImpl::Create(packed_call, qualified_name, schema)) {}

    ObjectPtr<FunctionImpl> pimpl_;
};

template<typename FType>
class TypedFunction;

template<typename R, typename... Args>
class TypedFunction<R(Args...)> : public ObjectRef {
public:
    using Self = TypedFunction;

    TypedFunction() = default;

    TypedFunction(std::nullopt_t) {}// NOLINT

    TypedFunction(Function packed_func) : packed_func_(std::move(packed_func)) {}//NOLINT

    template<typename FLambda,
             typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
    TypedFunction(const FLambda& callable, const QualifiedName& qualified_name = "no_qualified_name")
        : packed_func_(Function::FromTyped(callable, qualified_name)) {}

    template<typename FLambda,
             typename = std::enable_if_t<std::is_convertible_v<FLambda, std::function<R(Args...)>>>>
    Self& operator=(const FLambda& callable) {
        packed_func_ = Function::FromTyped(callable, "no_qualified_name");
        return *this;
    }

    Self& operator=(Function packed) {
        packed_func_ = std::move(packed);
        return *this;
    }

    operator Function() const {//NOLINT
        return packed();
    }

    AM_NODISCARD const Function& packed() const& {
        return packed_func_;
    }

    Function&& packed() && {
        return std::move(packed_func_);
    }

    AM_NODISCARD const String& schema() const noexcept {
        return packed_func_.schema();
    }

    R operator()(Args... args) const {
        if constexpr (std::is_same_v<R, void>) {
            packed_func_(std::forward<Args>(args)...);
            return;
        } else {
            Any res = packed_func_(std::forward<Args>(args)...);
            if constexpr (std::is_same_v<R, Any>) {
                return res;
            } else {
                return std::move(res).cast<R>();
            }
        }
    }

private:
    Function packed_func_;
};

}// namespace aethermind

#endif//AETHERMIND_FUNCTION_H
