//
// Created by 赵丹 on 2025/8/27.
//

#ifndef AETHERMIND_UTILS_THREAD_LOCAL_H
#define AETHERMIND_UTILS_THREAD_LOCAL_H

namespace aethermind {

template<typename T>
class ThreadLocal {
public:
    using Accessor = T* (*) ();
    explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

    ThreadLocal(const ThreadLocal&) = delete;
    ThreadLocal(ThreadLocal&&) noexcept = default;
    ThreadLocal& operator=(const ThreadLocal&) = delete;
    ThreadLocal& operator=(ThreadLocal&&) noexcept = default;
    ~ThreadLocal() = default;

    T& get() {
        return *accessor_();
    }

    T& operator*() {
        return get();
    }

    T* operator->() {
        return &get();
    }

private:
    Accessor accessor_;
};

#define DEFINE_TLS_STATIC(Type, Name)          \
    static aethermind::ThreadLocal<Type> Name( \
            []() {                             \
                static thread_local Type var;  \
                return &var;                   \
            })

#define DECLARE_TLS_CLASS_STATIC(Class, Type, Name) \
    static aethermind::ThreadLocal<Type> Name;

#define DEFINE_TLS_CLASS_STATIC(Class, Type, Name) \
    aethermind::ThreadLocal<Type> Class::Name(     \
            []() {                                 \
                static thread_local Type var;      \
                return &var;                       \
            })

}// namespace aethermind

#endif//AETHERMIND_UTILS_THREAD_LOCAL_H
