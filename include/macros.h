//
// Created by 赵丹 on 25-7-16.
//

#ifndef AETHERMIND_MACROS_H
#define AETHERMIND_MACROS_H

#ifdef __has_cpp_attribute
#if __has_cpp_attribute(nodiscard)
#define NODISCARD [[nodiscard]]
#else
#define NODISCARD
#endif

#if __has_cpp_attribute(maybe_unused)
#define MAYBE_UNUSED [[maybe_unused]]
#else
#define MAYBE_UNUSED
#endif
#endif

#if defined(__clang__)
#define __ubsan_ignore_pointer_overflow__ __attribute__((no_sanitize("pointer-overflow")))
#else
#define __ubsan_ignore_pointer_overflow__
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


#endif//AETHERMIND_MACROS_H
