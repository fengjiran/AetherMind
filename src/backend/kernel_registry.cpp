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

AM_NODISCARD inline Status ValidateResolveArgs(OpType op_type,
                                               const KernelSelector& selector) noexcept {
    if (op_type == OpType::kUnknown) {
        return Status::InvalidArgument("op_type cannot be kUnknown");
    }

    if (selector.device_type == DeviceType::kUndefined) {
        return Status::InvalidArgument("device_type cannot be kUndefined");
    }

    return Status::Ok();
}

}// namespace

KernelRegistry& KernelRegistry::Global() noexcept {
    static KernelRegistry registry;
    return registry;
}

Status KernelRegistry::RegisterGlobal(const KernelDescriptor& descriptor) {
    return Global().Register(descriptor);
}

Status KernelRegistry::Register(const KernelDescriptor& descriptor) {
    std::lock_guard<std::mutex> lock(mutex_);

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
    std::lock_guard<std::mutex> lock(mutex_);

    if (frozen_.load(std::memory_order_acquire)) {
        return;
    }
    BuildBucketIndex();
    frozen_.store(true, std::memory_order_release);
}

void KernelRegistry::BuildBucketIndex() noexcept {
    buckets_.clear();
    for (size_t i = 0; i < kernels_.size(); ++i) {
        buckets_[kernels_[i].op_type].push_back(i);
    }
}

StatusOr<const KernelDescriptor*> KernelRegistry::Resolve(OpType op_type,
                                                          const KernelSelector& selector) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!frozen_.load(std::memory_order_acquire)) {
        return Status(StatusCode::kFailedPrecondition,
                      "Cannot resolve kernel before registry has been frozen");
    }

    if (auto status = ValidateResolveArgs(op_type, selector); !status.ok()) {
        return status;
    }

    const KernelDescriptor* best = nullptr;

    if (const auto it = buckets_.find(op_type); it != buckets_.end()) {
        for (size_t idx: it->second) {
            const KernelDescriptor& descriptor = kernels_[idx];
            if (!SelectorMatches(descriptor.selector, selector)) {
                continue;
            }

            if (best == nullptr || descriptor.priority > best->priority) {
                best = &descriptor;
            }
        }
    }

    if (best == nullptr) {
        return Status::NotFound(
                "No matching kernel registered for op_type=" +
                std::string(ToString(op_type)) +
                ", selector=" + ToString(selector));
    }

    return best;
}

std::vector<const KernelDescriptor*> KernelRegistry::FindByOpType(OpType op_type) const {
    std::lock_guard<std::mutex> lock(mutex_);

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
    std::lock_guard<std::mutex> lock(mutex_);

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
