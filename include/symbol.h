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

using SymId = uint32_t;

// A Symbol is like an interned string, but with a little extra
// structure; it is namespaced via SymbolNamespace and the resulting
// intern pointers support efficient namespace testing.
class Symbol {
public:
    constexpr Symbol() : value_(0) {}
    explicit constexpr Symbol(SymId value) : value_(value) {}

    // Get a Symbol for a qualified string like "foo.bar.baz", "add.Tensor"
    static Symbol FromQualString(const String& qual);

    // Get a Symbol from a domain and an unqualified string like "aethermind.attr" and "bar"
    static Symbol FromDomainAndUnqualifiedName(const String& dom, const String& unqual);

    // Give a string corresponding to the qualified version of this name,
    // e.g., "aten::mm". This string format is made available to Python bindings
    // (so we know how to parse it.)
    NODISCARD String ToQualString() const;

    // Give a string corresponding to the unqualified version of this name, e.g.,
    // "mm". Use this in a context where the intended namespace of the string is
    // obvious; this is a *lossy* conversion.
    NODISCARD String ToUnQualString() const;

    NODISCARD String GetDomainString() const;

    NODISCARD Symbol NS() const;

    constexpr operator SymId() const { //NOLINT
        return value_;
    }

    // Constructors for our various namespaced strings. This will construct
    // the appropriate namespaced string, e.g., "attr::foo" for the
    // argument "foo", and then attempt to intern it.  DO NOT USE THIS
    // with a string literal; attr::foo should be available in that case
    // (and if it's not, you should add it to the built-ins list above.)
    static Symbol prim(const String& name);
    static Symbol cuda(const String& name);
    static Symbol attr(const String& name);

    NODISCARD bool IsPrim() const;
    NODISCARD bool IsCuda() const;
    NODISCARD bool IsAttr() const;

private:
    SymId value_;
};

inline bool operator==(Symbol lhs, Symbol rhs) noexcept {
    return static_cast<SymId>(lhs) == static_cast<SymId>(rhs);
}

}// namespace aethermind

namespace std {
template<>
struct hash<aethermind::Symbol> {
    size_t operator()(const aethermind::Symbol& sym) const noexcept {
        return std::hash<uint32_t>()(sym);
    }
};
}// namespace std

#endif//AETHERMIND_SYMBOL_H
