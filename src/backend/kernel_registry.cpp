#include "aethermind/backend/kernel_registry.h"

#include <ranges>
#include <string>

namespace aethermind {

namespace {

bool IsDuplicateRegistration(const KernelDescriptor& lhs,
                             const KernelDescriptor& rhs) noexcept {
    return lhs.op_type == rhs.op_type &&
           lhs.selector == rhs.selector;
}

}// namespace

Status KernelRegistry::Register(const KernelDescriptor& descriptor) {
    if (frozen_.load(std::memory_order_acquire)) {
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

void KernelRegistry::Freeze() noexcept {
    BuildBucketIndex();
    frozen_.store(true, std::memory_order_release);
}

void KernelRegistry::BuildBucketIndex() noexcept {
    for (size_t i = 0; i < kernels_.size(); ++i) {
        buckets_[kernels_[i].op_type].push_back(i);
    }
}

StatusOr<const KernelDescriptor*> KernelRegistry::Resolve(OpType op_type,
                                                          const KernelSelector& selector) const {
    return Resolve(KernelRequest{.op_type = op_type, .selector = selector});
}

StatusOr<const KernelDescriptor*> KernelRegistry::Resolve(const KernelRequest& request) const {
    if (!frozen_.load(std::memory_order_acquire)) {
        return Status(StatusCode::kFailedPrecondition,
                      "Cannot resolve kernel before registry has been frozen");
    }

    if (auto status = ValidateKernelRequest(request); !status.ok()) {
        return status;
    }

    const KernelDescriptor* best = nullptr;

    if (const auto it = buckets_.find(request.op_type); it != buckets_.end()) {
        for (size_t idx: it->second) {
            const KernelDescriptor& descriptor = kernels_[idx];
            if (!SelectorMatches(descriptor.selector, request.selector)) {
                continue;
            }

            if (best == nullptr || descriptor.priority > best->priority) {
                best = &descriptor;
            }
        }
    }

    if (best == nullptr) {
        return Status::NotFound(
                "No matching kernel registered: " + ToString(request));
    }

    return best;
}

std::vector<const KernelDescriptor*> KernelRegistry::FindByOpType(OpType op_type) const {
    std::vector<const KernelDescriptor*> result;
    if (const auto it = buckets_.find(op_type); it != buckets_.end()) {
        result.reserve(it->second.size());
        for (size_t idx: it->second) {
            result.push_back(&kernels_[idx]);
        }
    }
    return result;
}

std::string KernelRegistry::DebugDump() const {
    std::string out;
    for (size_t i = 0; i < kernels_.size(); ++i) {
        const auto& d = kernels_[i];
        if (i > 0) out += '\n';
        out += std::string(ToString(d.op_type)) + " | " +
               (d.name ? d.name : "<null>") + " | " +
               ToString(d.selector) + " | priority=" +
               std::to_string(d.priority);
    }
    return out;
}

}// namespace aethermind
