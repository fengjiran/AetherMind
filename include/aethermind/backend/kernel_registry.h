//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_REGISTRY_H
#define AETHERMIND_BACKEND_KERNEL_REGISTRY_H

#include "aethermind/backend/kernel_descriptor.h"
#include "aethermind/base/kernel_selector.h"
#include "aethermind/base/status.h"
#include "utils/hash.h"

#include <atomic>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace aethermind {

struct RegistrationKey {
    OpType op_type = OpType::kUnknown;
    KernelSelector selector{};
    CpuFeatureSet cpu_requirements{};

    friend bool operator==(const RegistrationKey& lhs, const RegistrationKey& rhs) noexcept {
        return lhs.op_type == rhs.op_type &&
               lhs.selector == rhs.selector &&
               lhs.cpu_requirements == rhs.cpu_requirements;
    }
};

struct RegistrationKeyHash {
    std::size_t operator()(const RegistrationKey& key) const noexcept {
        std::size_t seed = 0;
        seed = hash_combine(seed, std::hash<OpType>{}(key.op_type));
        seed = hash_combine(seed, std::hash<KernelSelector>{}(key.selector));
        seed = hash_combine(seed, key.cpu_requirements.Hash());
        return seed;
    }
};

class KernelRegistry {
public:
    static KernelRegistry& Global() noexcept;

    Status Register(const KernelDescriptor& descriptor);

    /// Returns every descriptor whose structural selector can serve `selector`.
    /// Callers must apply backend-specific execution requirements before they
    /// choose a concrete kernel.
    AM_NODISCARD StatusOr<std::vector<const KernelDescriptor*>> FindCandidates(
            OpType op_type,
            const KernelSelector& selector) const;

    AM_NODISCARD Status Freeze();

    AM_NODISCARD size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return kernels_.size();
    }

    AM_NODISCARD bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return kernels_.empty();
    }

    AM_NODISCARD bool frozen() const noexcept {
        return frozen_.load(std::memory_order_acquire);
    }

    /// Debug inspection API; unlike FindCandidates, this does not apply a
    /// structural selector filter.
    AM_NODISCARD StatusOr<std::vector<const KernelDescriptor*>> FindByOpType(OpType op_type) const;

    AM_NODISCARD std::string DebugDump() const;

private:
    AM_NODISCARD Status BuildBucketIndex();

    mutable std::mutex mutex_{};
    std::atomic<bool> frozen_{false};
    std::vector<KernelDescriptor> kernels_{};
    std::unordered_map<OpType, std::vector<size_t>> buckets_{};
    std::unordered_set<RegistrationKey, RegistrationKeyHash> registration_keys_{};
};

} // namespace aethermind
#endif
