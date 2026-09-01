#ifndef AETHERMIND_UTILS_THREAD_LOCAL_H
#define AETHERMIND_UTILS_THREAD_LOCAL_H

/// @file
/// @brief Access to lazily initialized thread-local objects.

namespace aethermind {

/// @brief Provides access to a thread-local object through a pointer accessor.
///
/// The wrapper does not own the returned object. The accessor is invoked for
/// every access and must return a valid pointer for the current thread.
///
/// @tparam T Type of the thread-local object.
/// @note The wrapper does not synchronize access to the object returned by the
///       accessor.
template<typename T>
class ThreadLocal {
public:
    /// @brief Function that returns the current thread's object.
    using Accessor = T* (*) ();

    /// @brief Creates a wrapper around an accessor.
    /// @param accessor Function that returns a non-null object pointer.
    /// @pre `accessor` is not null.
    explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

    ThreadLocal(const ThreadLocal&) = delete;
    ThreadLocal(ThreadLocal&&) noexcept = default;
    ThreadLocal& operator=(const ThreadLocal&) = delete;
    ThreadLocal& operator=(ThreadLocal&&) noexcept = default;
    ~ThreadLocal() = default;

    /// @brief Returns the current thread's object.
    /// @return A reference to the object returned by the accessor.
    T& get() {
        return *accessor_();
    }

    /// @brief Dereferences the current thread's object.
    /// @return A reference to the current thread's object.
    T& operator*() {
        return get();
    }

    /// @brief Accesses a member of the current thread's object.
    /// @return A pointer to the current thread's object.
    T* operator->() {
        return &get();
    }

private:
    Accessor accessor_;
};

/// @brief Defines a namespace-scope `ThreadLocal` with one object per thread.
///
/// @param Type Thread-local object type.
/// @param Name Name of the wrapper variable.
/// @note The generated object has internal linkage. The accessor creates the
///       thread-local object on first use in each thread.
#define DEFINE_TLS_STATIC(Type, Name)          \
    static aethermind::ThreadLocal<Type> Name( \
            []() {                             \
                static thread_local Type var;  \
                return &var;                   \
            })

/// @brief Declares a static `ThreadLocal` member for a class.
///
/// @param Class Class that owns the static member.
/// @param Type Thread-local object type.
/// @param Name Static member name.
/// @note Pair this declaration with one `DEFINE_TLS_CLASS_STATIC` definition.
#define DECLARE_TLS_CLASS_STATIC(Class, Type, Name) \
    static aethermind::ThreadLocal<Type> Name;

/// @brief Defines a static class member with one object per thread.
///
/// @param Class Class that owns the static member.
/// @param Type Thread-local object type.
/// @param Name Static member name.
/// @note Provide exactly one definition for each declaration in a source file.
#define DEFINE_TLS_CLASS_STATIC(Class, Type, Name) \
    aethermind::ThreadLocal<Type> Class::Name(     \
            []() {                                 \
                static thread_local Type var;      \
                return &var;                       \
            })

} // namespace aethermind

#endif // AETHERMIND_UTILS_THREAD_LOCAL_H
