//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_REGISTRY_H
#define AETHERMIND_BACKEND_KERNEL_REGISTRY_H

#include "aethermind/backend/kernel_descriptor.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aethermind {

struct RegistrationKey {
    OpType op_type;
    KernelSelector selector;

    friend bool operator==(const RegistrationKey& lhs, const RegistrationKey& rhs) noexcept {
        return lhs.op_type == rhs.op_type && lhs.selector == rhs.selector;
    }
};

struct RegistrationKeyHash {
    std::size_t operator()(const RegistrationKey& key) const noexcept {
        const auto h1 = std::hash<OpType>{}(key.op_type);
        const auto h2 = std::hash<KernelSelector>{}(key.selector);
        return h1 ^ (h2 << 1);
    }
};

class KernelRegistry {
public:
    static KernelRegistry& Global() noexcept;

    Status Register(const KernelDescriptor& descriptor);

    AM_NODISCARD StatusOr<const KernelDescriptor*> Resolve(OpType op_type,
                                                           const KernelSelector& selector) const;

    void Freeze();

    AM_NODISCARD size_t size() const noexcept {
        return kernels_.size();
    }

    AM_NODISCARD bool empty() const noexcept {
        return kernels_.empty();
    }

    AM_NODISCARD bool frozen() const noexcept {
        return frozen_.load(std::memory_order_acquire);
    }

    AM_NODISCARD std::vector<const KernelDescriptor*> FindByOpType(OpType op_type) const;

    AM_NODISCARD std::string DebugDump() const;

private:
    void BuildBucketIndex();

    mutable std::mutex mutex_{};
    std::atomic<bool> frozen_{false};
    std::vector<KernelDescriptor> kernels_{};
    std::unordered_map<OpType, std::vector<size_t>> buckets_{};
    std::unordered_set<RegistrationKey, RegistrationKeyHash> registration_keys_{};
};

}// namespace aethermind
#endif
