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

private:
    static void RegisterFunc(const String& name, const String& doc, Function func, bool allow_override);
};

}// namespace aethermind

#endif//AETHERMIND_REGISTRY_H
