//
// Created by 赵丹 on 25-7-16.
//

#ifndef AETHERMIND_MACROS_H
#define AETHERMIND_MACROS_H

#if __cplusplus >= 202002L
#define CPP20
#endif

#ifdef __has_feature
#define AM_HAS_FEATURE(...) __has_feature(__VA_ARGS__)
#else
#define AM_HAS_FEATURE(...) 0
#endif

#ifdef __has_builtin
#define AM_HAS_BUILTIN(...) __has_builtin(__VA_ARGS__)
#else
#define AM_HAS_BUILTIN(...) 0
#endif

#ifdef __has_cpp_attribute
#define AM_HAS_CPP_ATTRIBUTE(...) __has_cpp_attribute(__VA_ARGS__)
#else
#define AM_HAS_CPP_ATTRIBUTE(...) 0
#endif

#if AM_HAS_CPP_ATTRIBUTE(nodiscard)
#define AM_NODISCARD [[nodiscard]]
#else
#define AM_NODISCARD
#endif

#if AM_HAS_CPP_ATTRIBUTE(maybe_unused)
#define AM_MAYBE_UNUSED [[maybe_unused]]
#else
#define AM_MAYBE_UNUSED
#endif

#if AM_HAS_CPP_ATTRIBUTE(likely)
#define AM_LIKELY [[likely]]
#else
#define AM_LIKELY
#endif

#if AM_HAS_CPP_ATTRIBUTE(unlikely)
#define AM_UNLIKELY [[unlikely]]
#else
#define AM_UNLIKELY
#endif

#ifndef AM_SANITIZE_ADDRESS
#if AM_HAS_FEATURE(address_sanitizer) || \
        defined(__SANITIZE_ADDRESS__) || \
        AM_HAS_FEATURE(hwaddress_sanitizer)
#define AM_SANITIZE_ADDRESS 1
#endif
#endif

/**
 * Define a convenience macro to test when ASAN, UBSAN, TSAN or MSAN sanitizer
 * are being used
 */
#ifndef AM_SANITIZE
#if defined(AM_SANITIZE_ADDRESS) || defined(AM_SANITIZE_THREAD) ||      \
        defined(AM_SANITIZE_MEMORY) || defined(AM_SANITIZE_DATAFLOW) || \
        defined(AM_SANITIZE_UNDEFINED_BEHAVIOR)
#define AM_SANITIZE 1
#endif
#endif

#define AETHERMIND_THROW_EXCEPTION noexcept(false)

#if defined(__clang__)
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#define __ubsan_ignore_float_divide_by_zero__ __attribute__((no_sanitize("float-divide-by-zero")))
#else
#define __ubsan_ignore_pointer_overflow__
#define __ubsan_ignore_float_divide_by_zero__
#endif

#if defined(__GNUC__)
#define ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define ATTRIBUTE_UNUSED
#endif


#define UNUSED(expr)   \
    do {               \
        (void) (expr); \
    } while (false)

#define STR_CONCAT_(__x, __y) __x##__y
#define STR_CONCAT(__x, __y) STR_CONCAT_(__x, __y)

#define REG_VAR_DEF static ATTRIBUTE_UNUSED uint32_t _make_unique_tid_

#if defined(__GNUC__) || defined(__clang__)
#define FUNC_SIG __PRETTY_FUNCTION__
#elif defined(_MSC_VER)
#define FUNC_SIG __FUNCSIG__
#else
#define FUNC_SIG __func__
#endif

#define AM_UNREACHABLE() __builtin_unreachable()

#define DEFINE_OBJECT_NULLTYPE(ClassName)                \
    class ClassName##NullType final : public ClassName { \
        ClassName##NullType() = default;                 \
                                                         \
    public:                                              \
        static ClassName* singleton() noexcept {         \
            static ClassName##NullType singleton_;       \
            return &singleton_;                          \
        }                                                \
    };

#define DEFINE_STATIC_FUNCTION_(FuncName)       \
    static void FuncName();                     \
    STR_CONCAT(REG_VAR_DEF, __COUNTER__) = [] { \
        FuncName();                             \
        return 0;                               \
    }();                                        \
    static void FuncName()

#define DEFINE_STATIC_FUNCTION() DEFINE_STATIC_FUNCTION_(STR_CONCAT(_static_func_tid_, __COUNTER__))
#endif//AETHERMIND_MACROS_H
