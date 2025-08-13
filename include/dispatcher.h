//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_DISPATCHER_H
#define AETHERMIND_DISPATCHER_H

#include "operator_name.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace aethermind {

class OperatorSchema {
public:
    OperatorSchema(std::string name, std::string overload_name)
        : name_({std::move(name), std::move(overload_name)}) {}

private:
    OperatorName name_;
};

class OperatorEntry {
public:
private:
    OperatorName name_;
};

class Dispatcher {
public:
    static Dispatcher& Global() {
        static Dispatcher inst;
        return inst;
    }

    std::vector<OperatorName> ListAllOpNames();

private:
    Dispatcher() = default;

    std::unordered_map<OperatorName, std::unique_ptr<OperatorEntry>> table_;
};

}// namespace aethermind

#endif//AETHERMIND_DISPATCHER_H
