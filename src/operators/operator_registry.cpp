#include "aethermind/operators/operator_registry.h"

#include <utility>

namespace aethermind {

bool OperatorRegistry::Register(OpType op_type, FactoryFunc factory) {
    if (op_type == OpType::kUnknown || !factory) {
        return false;
    }

    auto& registry = Registry();
    return registry.emplace(op_type, std::move(factory)).second;
}

StatusOr<std::unique_ptr<Operator>> OperatorRegistry::Create(OpType op_type, const std::any& params) {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("OperatorRegistry cannot create kUnknown operator");
    }

    const auto& registry = Registry();
    const auto it = registry.find(op_type);
    if (it == registry.end()) {
        return Status::NotFound(
                "No operator factory registered for " + std::string(ToString(op_type)));
    }

    return it->second(params);
}

std::unordered_map<OpType, OperatorRegistry::FactoryFunc>& OperatorRegistry::Registry() {
    static std::unordered_map<OpType, FactoryFunc> registry;
    return registry;
}

}// namespace aethermind
