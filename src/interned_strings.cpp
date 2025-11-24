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
#define SYMBOL_ENTRY(n, s) {#n, #s, n::s, namespaces::n},
        FORALL_NS_SYMBOLS(SYMBOL_ENTRY)
#undef SYMBOL_ENTRY
};

}// namespace

InternedStrings::InternedStrings() : symbol_infos_(static_cast<size_t>(keys::num_symbols)) {
    for (const auto& entry: entries) {
        auto qual_name = GetQualNameFromEntry(entry);
        string_to_symbol_[qual_name] = entry.sym;
        symbol_infos_[entry.sym] = {entry.ns_sym, std::move(qual_name), entry.unqual_name};
    }
}

Symbol InternedStrings::_symbol(const String& s) {
    if (auto it = string_to_symbol_.find(s); it != string_to_symbol_.end()) {
        return it->second;
    }

    auto pos = s.find("::");
    CHECK(pos != String::npos) << "all symbols must have a namespace, <namespace>::<string>, but found: " << s;
    Symbol ns = _symbol("namespaces::" + s.substr(0, pos));
    Symbol sym(symbol_infos_.size());
    string_to_symbol_[s] = sym;
    symbol_infos_.push_back({ns, s, s.substr(pos + 2)});
    return sym;
}


Symbol InternedStrings::symbol(const String& s) {
}


}// namespace aethermind