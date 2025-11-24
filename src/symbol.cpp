//
// Created by richard on 11/24/25.
//

#include "symbol.h"
#include "interned_strings.h"

namespace aethermind {

const String& GetDomainPrefix() {
    static const String domain_prefix = "aethermind";
    return domain_prefix;
}

Symbol Symbol::FromQualString(const String& qual) {
    return InternedStrings::Global().symbol(qual);
}


}// namespace aethermind