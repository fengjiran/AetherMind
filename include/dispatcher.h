//
// Created by 赵丹 on 2025/8/13.
//

#ifndef AETHERMIND_DISPATCHER_H
#define AETHERMIND_DISPATCHER_H

// ============================================================================
// FROZEN / DEPRECATED NOTICE
// ============================================================================
// This file is part of the legacy dispatch system and is NO LONGER the main
// dispatch path for new operator implementations.
//
// New dispatch mainline (since Batch 1-3 of dispatch redesign):
//   - OpType + KernelSelector + KernelDescriptor
//   - Backend-owned KernelRegistry
//   - Plan-build-time resolve via ExecutionPlanBuilder
//   - Executor direct call on frozen ResolvedKernel
//
// This file is kept for migration compatibility only and should NOT be
// extended with new runtime dispatch responsibilities. All new operators
// must use the OpType-centered dispatch path.
//
// See: docs/designs/dispatch_design.md for the new mainline architecture.
// ============================================================================

#include "operator_name.h"

#include <memory>
#include <unordered_map>
#include <vector>

namespace aethermind {

class OperatorSchema {
public:
    OperatorSchema(String name, String overload_name)
        : name_(static_cast<std::string>(name), static_cast<std::string>(overload_name)) {}

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

#endif// AETHERMIND_DISPATCHER_H
