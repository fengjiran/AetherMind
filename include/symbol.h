//
// Created by richard on 11/24/25.
//

#ifndef AETHERMIND_SYMBOL_H
#define AETHERMIND_SYMBOL_H

#include "container/string.h"

#include <cstdint>
#include <functional>

namespace aethermind {

// aethermind
const String& GetDomainPrefix();

// A Symbol is like an interned string, but with a little extra
// structure; it is namespaced via SymbolNamespace and the resulting
// intern pointers support efficient namespace testing.
class Symbol {
public:
    constexpr Symbol() : value_(0) {}
    explicit constexpr Symbol(uint32_t value) : value_(value) {}

    // Get a Symbol for a qualified string like "foo.bar.baz", "add.Tensor"
    static Symbol FromQualString(const String& qual);

    // Give a string corresponding to the qualified version of this name,
    // e.g., "aten::mm". This string format is made available to Python bindings
    // (so we know how to parse it.)


    constexpr operator uint32_t() const {
        return value_;
    }

private:
    uint32_t value_;
};

}// namespace aethermind

#endif//AETHERMIND_SYMBOL_H
