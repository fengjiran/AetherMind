//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_REGISTRY_H
#define AETHERMIND_BACKEND_KERNEL_REGISTRY_H

#include "aethermind/backend/kernel_descriptor.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"

#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace aethermind {

class KernelRegistry {
public:
    Status Register(const KernelDescriptor& descriptor);

    AM_NODISCARD StatusOr<const KernelDescriptor*> Resolve(OpType op_type,
                                                           const KernelSelector& selector) const;

    void Freeze() noexcept;

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
    void BuildBucketIndex() noexcept;

    std::atomic<bool> frozen_{false};
    std::vector<KernelDescriptor> kernels_{};
    std::unordered_map<OpType, std::vector<size_t>> buckets_{};
};

}// namespace aethermind
#endif
