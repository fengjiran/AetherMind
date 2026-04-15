#include "aethermind/backend/kernel_registry.h"

namespace aethermind {

void KernelRegistry::Register(const KernelKey& key, KernelFunc fn) {
    AM_CHECK(fn != nullptr, "Kernel function cannot be null");
    AM_CHECK(!kernels_.contains(key),
             "Duplicate kernel registration for operator: {}",
             ToString(key.op_name).c_str());
    kernels_.emplace(key, fn);
}

KernelFunc KernelRegistry::Find(const KernelKey& key) const noexcept {
    const auto it = kernels_.find(key);
    if (it == kernels_.end()) {
        return nullptr;
    }
    return it->second;
}

}// namespace aethermind
