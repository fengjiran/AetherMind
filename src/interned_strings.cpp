//
// Created by richard on 11/24/25.
//

#include "interned_strings.h"

namespace aethermind {

namespace {

struct Entry {
    const String ns; // namespace
    const String unqual_name;
    const Symbol sym;
    const Symbol ns_sym;
};

String GetQualNameFromEntry(const Entry& entry) {
    const String sep = "::";
    const auto ns_len = entry.ns.size();
    const auto sep_len = sep.size();
    const auto unqual_name_len = entry.unqual_name.size();

    String s;
    s.reserve(ns_len + sep_len + unqual_name_len);
    s += entry.ns;
    s += sep;
    s += entry.unqual_name;
    return s;
}

// constexpr Entry entries[] = {
//
// };

}

}