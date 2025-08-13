//
// Created by 赵丹 on 2025/8/13.
//

#include "dispatcher.h"

namespace aethermind {

std::vector<OperatorName> Dispatcher::ListAllOpNames() {
    std::vector<OperatorName> names;
    names.reserve(table_.size());
    for (const auto& [name, _]: table_) {
        names.emplace_back(name);
    }
    return names;
}


}// namespace aethermind
