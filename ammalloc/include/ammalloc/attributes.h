// Attribute and builtin macros used by ammalloc.
//
// Self-contained counterpart of include/macros.h: defines only the
// handful of macros ammalloc actually uses, so that ammalloc's public
// headers do not depend on the AetherMind include tree and can be
// built/distributed standalone. Kept in sync with include/macros.h by
// intent.

#ifndef AMMALLOC_ATTRIBUTES_H
#define AMMALLOC_ATTRIBUTES_H

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

#if AM_HAS_BUILTIN(__builtin_prefetch)
#define AM_BUILTIN_PREFETCH(...) __builtin_prefetch(__VA_ARGS__)
#else
#define AM_BUILTIN_PREFETCH(...)
#endif

#if AM_HAS_CPP_ATTRIBUTE(nodiscard)
#define AM_NODISCARD [[nodiscard]]
#else
#define AM_NODISCARD
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

#if defined(__GNUC__) || defined(__clang__)
#define AM_NOINLINE __attribute__((noinline))
#define AM_ALWAYS_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define AM_NOINLINE __declspec(noinline)
#define AM_ALWAYS_INLINE __forceinline
#else
#define AM_NOINLINE
#define AM_ALWAYS_INLINE inline
#endif

#endif// AMMALLOC_ATTRIBUTES_H