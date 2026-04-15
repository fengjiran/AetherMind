#include "aethermind/backend/kernel_registry.h"

#include <algorithm>

namespace aethermind {

namespace {

bool IsDuplicateRegistration(const KernelDescriptor& lhs,
                             const KernelDescriptor& rhs) noexcept {
    return lhs.op_type == rhs.op_type &&
           lhs.selector == rhs.selector;
}

}// namespace

Status KernelRegistry::Register(const KernelDescriptor& descriptor) {
    if (!IsValidKernelDescriptor(descriptor)) {
        return Status::InvalidArgument("Kernel function cannot be null");
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

Status KernelRegistry::Resolve(OpType op_type,
                               const KernelSelector& selector,
                               const KernelDescriptor** out) const noexcept {
    if (out == nullptr) {
        return Status::InvalidArgument("Output descriptor pointer cannot be null");
    }

    *out = nullptr;

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
        return Status::NotFound("No matching kernel registered");
    }

    *out = best;
    return Status::Ok();
}

}// namespace aethermind
