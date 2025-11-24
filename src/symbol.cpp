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
    CHECK(dom.compare(0, GetDomainPrefix().size(), GetDomainPrefix()) == 0)
            << "Symbol: domain string is expected to be prefixed with '" << GetDomainPrefix()
            << "', e.g. 'aethermind.add'";
    auto qual_name = dom.substr(GetDomainPrefix().size()) + "::" + unqual;
    return FromQualString(qual_name);
}

String Symbol::ToQualString() const {
    return InternedStrings::Global().string(*this).first;
}

String Symbol::ToUnQualString() const {
    return InternedStrings::Global().string(*this).second;
}

Symbol Symbol::NS() const {
    return InternedStrings::Global().NS(*this);
}


}// namespace aethermind