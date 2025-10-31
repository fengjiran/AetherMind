//
// Created by richard on 10/31/25.
//

#ifndef AETHERMIND_UTILS_QUALIFIED_NAME_H
#define AETHERMIND_UTILS_QUALIFIED_NAME_H

#include "container/array_view.h"
#include "container/string.h"

namespace aethermind {

// Represents a name of the form "foo.bar.baz"
class QualifiedName {
public:
    QualifiedName() = default;

private:
    // The actual list of names, like "{foo, bar, baz}"
    std::vector<String> atoms_;

    // Cached accessors, derived from `atoms_`.
    String qualified_name_;
    String name_;
    String prefix_;
};

}// namespace aethermind

// hash function
namespace std {

}

#endif//AETHERMIND_UTILS_QUALIFIED_NAME_H
