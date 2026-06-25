#include "aethermind/operators/operator_registry.h"

#include <cstdio>
#include <cstdlib>
#include <utility>

namespace aethermind {

Status OperatorRegistry::Register(OpType op_type, FactoryFunc factory) {
    return Register(op_type, Descriptor{.factory_ = std::move(factory)});
}

Status OperatorRegistry::Register(OpType op_type, Descriptor descriptor) {
    if (op_type == OpType::kUnknown || !descriptor.factory_) {
        return Status::InvalidArgument("OperatorRegistry requires a known OpType and non-null factory");
    }

    std::lock_guard<std::mutex> lock(Mutex());
    auto& registry = Registry();
    if (!registry.emplace(op_type, std::move(descriptor)).second) {
        return Status::AlreadyExists(
                "Operator factory already registered for " + std::string(ToString(op_type)));
    }
    return Status::Ok();
}

bool OperatorRegistry::RegisterOrAbort(OpType op_type, FactoryFunc factory, const char* op_name) {
    return RegisterOrAbort(op_type, Descriptor{.factory_ = std::move(factory)}, op_name);
}

bool OperatorRegistry::RegisterOrAbort(OpType op_type, Descriptor descriptor, const char* op_name) {
    const Status status = Register(op_type, std::move(descriptor));
    if (status.ok()) {
        return true;
    }

    std::fprintf(stderr,
                 "Failed to register operator %s: %s\n",
                 op_name != nullptr ? op_name : "<unknown>",
                 status.ToString().c_str());
    std::abort();
}

StatusOr<std::unique_ptr<Operator>> OperatorRegistry::Create(OpType op_type, const OpParams& params) {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("OperatorRegistry cannot create kUnknown operator");
    }

    FactoryFunc factory;
    {
        std::lock_guard<std::mutex> lock(Mutex());
        const auto& registry = Registry();
        const auto it = registry.find(op_type);
        if (it == registry.end()) {
            return Status::NotFound(
                    "No operator factory registered for " + std::string(ToString(op_type)));
        }
        factory = it->second.factory_;
    }

    return factory(params);
}

StatusOr<OpParams> OperatorRegistry::CreateDefaultParams(OpType op_type) {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("OperatorRegistry cannot create default params for kUnknown operator");
    }

    ParamFactoryFunc make_default_params;
    {
        std::lock_guard<std::mutex> lock(Mutex());
        const auto& registry = Registry();
        const auto it = registry.find(op_type);
        if (it == registry.end()) {
            return Status::NotFound(
                    "No operator descriptor registered for " + std::string(ToString(op_type)));
        }
        make_default_params = it->second.make_default_params_;
    }

    if (!make_default_params) {
        return Status::NotFound(
                "No default params factory registered for " + std::string(ToString(op_type)));
    }
    return make_default_params();
}

std::unordered_map<OpType, OperatorRegistry::Descriptor>& OperatorRegistry::Registry() {
    static std::unordered_map<OpType, Descriptor> registry;
    return registry;
}

std::mutex& OperatorRegistry::Mutex() {
    static std::mutex mutex;
    return mutex;
}

}// namespace aethermind
