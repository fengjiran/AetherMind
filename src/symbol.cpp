//
// Created by richard on 11/24/25.
//

#include "symbol.h"
#include "interned_strings.h"

namespace aethermind {

const String& GetDomainPrefix() {
    static const String domain_prefix = "aethermind.";
    return domain_prefix;
}

Symbol Symbol::FromQualString(const String& qual_name) {
    return InternedStrings::Global().GetSymbol(qual_name);
}

Symbol Symbol::FromDomainAndUnqualifiedName(const String& dom, const String& unqual) {
    AM_CHECK(dom.compare(0, GetDomainPrefix().size(), GetDomainPrefix()) == 0,
             "Symbol: domain string is expected to be prefixed with '{}', e.g. 'aethermind.add'", GetDomainPrefix().c_str());
    auto qual_name = dom.substr(GetDomainPrefix().size()) + "::" + unqual;
    return FromQualString(qual_name);
}

String Symbol::ToQualString() const {
    return InternedStrings::Global().string(*this).first;
}

String Symbol::ToUnQualString() const {
    return InternedStrings::Global().string(*this).second;
}

String Symbol::GetDomainString() const {
    return GetDomainPrefix() + NS().ToUnQualString();
}

Symbol Symbol::NS() const {
    return InternedStrings::Global().NS(*this);
}

Symbol Symbol::prim(const String& name) {
    return FromQualString("prim::" + name);
}

Symbol Symbol::cuda(const String& name) {
    return FromQualString("cuda::" + name);
}

Symbol Symbol::attr(const String& name) {
    return FromQualString("attr::" + name);
}

bool Symbol::IsPrim() const {
    return NS() == namespaces::prim;
}

bool Symbol::IsCuda() const {
    return NS() == namespaces::cuda;
}

bool Symbol::IsAttr() const {
    return NS() == namespaces::attr;
}

}// namespace aethermind