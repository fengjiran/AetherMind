#include "aethermind/backend/kernel_registry.h"

#include <string>
#include <ranges>

namespace aethermind {

namespace {

bool IsDuplicateRegistration(const KernelDescriptor& lhs,
                             const KernelDescriptor& rhs) noexcept {
    return lhs.op_type == rhs.op_type &&
           lhs.selector == rhs.selector;
}

}// namespace

Status KernelRegistry::Register(const KernelDescriptor& descriptor) {
    if (frozen_) {
        return Status(StatusCode::kFailedPrecondition,
                      "Cannot register kernel after registry has been frozen");
    }

    if (auto status = ValidateKernelDescriptor(descriptor); !status.ok()) {
        return status;
    }

    const auto duplicate = std::ranges::find_if(kernels_,
                                                [&](const KernelDescriptor& existing) {
                                                    return IsDuplicateRegistration(existing, descriptor);
                                                });

    if (duplicate != kernels_.end()) {
        return Status(StatusCode::kAlreadyExists, "Duplicate kernel registration");
    }

    kernels_.push_back(descriptor);
    return Status::Ok();
}

Status KernelRegistry::Freeze() noexcept {
    frozen_ = true;
    return Status::Ok();
}

StatusOr<const KernelDescriptor*> KernelRegistry::Resolve(
        OpType op_type,
        const KernelSelector& selector) const {
    if (!frozen_) {
        return Status(StatusCode::kFailedPrecondition,
                      "Cannot resolve kernel before registry has been frozen");
    }

    const KernelDescriptor* best = nullptr;
    for (const KernelDescriptor& descriptor: kernels_) {
        if (descriptor.op_type != op_type) {
            continue;
        }

        if (!SelectorMatches(descriptor.selector, selector)) {
            continue;
        }

        if (best == nullptr || descriptor.priority > best->priority) {
            best = &descriptor;
        }
    }

    if (best == nullptr) {
        return Status::NotFound(
                "No matching kernel registered: op_type=" +
                std::string(ToString(op_type)) +
                ", selector=" + ToString(selector));
    }

    return best;
}

}// namespace aethermind
