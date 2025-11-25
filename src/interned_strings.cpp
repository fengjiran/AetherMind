//
// Created by richard on 11/24/25.
//

#include "interned_strings.h"

namespace aethermind {

namespace {

struct Entry {
    const char* const ns;// namespace
    const char* const unqual_name;
    const Symbol sym;
    const Symbol ns_sym;
};

String GetQualNameFromEntry(const Entry& entry) {
    const char* const sep = "::";
    const auto ns_len = std::strlen(entry.ns);
    const auto sep_len = std::strlen(sep);
    const auto unqual_name_len = std::strlen(entry.unqual_name);

    String s;
    s.reserve(ns_len + sep_len + unqual_name_len);
    s += entry.ns;
    s += sep;
    s += entry.unqual_name;
    return s;
}

constexpr Entry entries[] = {
#define SYMBOL_ENTRY(n, s) Entry{#n, #s, n::s, namespaces::n},
        FORALL_NS_SYMBOLS(SYMBOL_ENTRY)
#undef SYMBOL_ENTRY
};

}// namespace

InternedStrings::InternedStrings() : symbol_infos_(static_cast<size_t>(Keys::num_symbols)) {
    for (const auto& entry: entries) {
        auto qual_name = GetQualNameFromEntry(entry);
        string_to_symbol_[qual_name] = entry.sym;
        symbol_infos_[entry.sym] = SymbolInfo{entry.ns_sym, std::move(qual_name), entry.unqual_name};
    }
}

Symbol InternedStrings::GetSymbolImpl(const String& s) {
    if (const auto it = string_to_symbol_.find(s); it != string_to_symbol_.end()) {
        return it->second;
    }

    auto pos = s.find("::");
    CHECK(pos != String::npos) << "all symbols must have a namespace, <namespace>::<string>, but found: " << s;
    const Symbol ns = GetSymbolImpl("namespaces::" + s.substr(0, pos));
    Symbol sym(symbol_infos_.size());
    string_to_symbol_[s] = sym;
    // symbol_infos_.push_back({ns, s, s.substr(pos + 2)});
    symbol_infos_.emplace_back(ns, s, s.substr(pos + 2));
    return sym;
}

Symbol InternedStrings::GetSymbol(const String& s) {
    std::lock_guard lock(mutex_);
    return GetSymbolImpl(s);
}

std::pair<String, String> InternedStrings::string(Symbol sym) {
    switch (sym) {
#define CASE(ns, s)                    \
    case static_cast<SymId>(ns::s): \
        return {#ns "::" #s, #s};
        FORALL_NS_SYMBOLS(CASE)
#undef CASE
        default: {
            std::lock_guard lock(mutex_);
            const auto& sym_info = symbol_infos_[sym];
            return {sym_info.qual_name, sym_info.unqual_name};
        }
    }
}

Symbol InternedStrings::NS(Symbol sym) {
    switch (sym) {
#define CASE(ns, s)                    \
    case static_cast<SymId>(ns::s): \
        return namespaces::ns;
        FORALL_NS_SYMBOLS(CASE)
#undef CASE
        default: {
            std::lock_guard lock(mutex_);
            return symbol_infos_[sym].ns;
        }
    }
}

std::vector<String> InternedStrings::ListAllSymbolNames() const {
    std::vector<String> res;
    res.reserve(symbol_infos_.size());
    for (const auto& sym_info: symbol_infos_) {
        res.push_back(sym_info.qual_name);
    }
    return res;
}

}// namespace aethermind