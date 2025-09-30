//
// Created by richard on 9/29/25.
//

#ifndef AETHERMIND_REGISTRY_H
#define AETHERMIND_REGISTRY_H

#include "function.h"

namespace aethermind {

class Registry {
public:
    template<typename F>
    Registry& def(const String& name, const String& doc, F&& func) {
        RegisterFunc(name, doc, Function::FromTyped(std::forward<F>(func)), false);
        return *this;
    }

    template<typename F>
    Registry& def_packed(const String& name, const String& doc, F&& func) {
        RegisterFunc(name, doc, Function::FromPacked(std::forward<F>(func)), false);
        return *this;
    }

    template<typename F>
    Registry& def_method(const String& name, const String& doc, F&& func) {
        RegisterFunc(name, doc, GetMethod(name, std::forward<F>(func)));
        return *this;
    }

private:
    static void RegisterFunc(const String& name, const String& doc, const Function& func, bool allow_override);

    template<typename Class, typename R, typename... Args>
    static Function GetMethod(const String& name, R (Class::*func)(Args...)) {
        static_assert(std::is_base_of_v<Object, Class>, "Class must be derived from Object");
        auto f = [func](const Class* target, Args... args) -> R {
            return (const_cast<Class*>(target)->*func)(std::forward<Args>(args)...);
        };
        return Function::FromTyped(f, name);
    }

    template<typename Class, typename R, typename... Args>
    static Function GetMethod(const String& name, R (Class::*func)(Args...) const) {
        static_assert(std::is_base_of_v<Object, Class>, "Class must be derived from Object");
        auto f = [func](const Class* target, Args... args) -> R {
            return (target->*func)(std::forward<Args>(args)...);
        };
        return Function::FromTyped(f, name);
    }

    template<typename F>
    static Function GetMethod(const String& name, F&& func) {
        return Function::FromTyped(std::forward<F>(func), name);
    }
};

}// namespace aethermind

#endif//AETHERMIND_REGISTRY_H
