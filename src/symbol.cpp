//
// Created by richard on 11/24/25.
//

#include "symbol.h"

namespace aethermind {

const String& GetDomainPrefix() {
    static const String domain_prefix = "aethermind";
    return domain_prefix;
}


}// namespace aethermind