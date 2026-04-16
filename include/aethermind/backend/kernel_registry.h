//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_REGISTRY_H
#define AETHERMIND_BACKEND_KERNEL_REGISTRY_H

#include "aethermind/backend/kernel_descriptor.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/status.h"

#include <vector>

namespace aethermind {

class KernelRegistry {
public:
    Status Register(const KernelDescriptor& descriptor);

    StatusOr<const KernelDescriptor*> Resolve(
            OpType op_type,
            const KernelSelector& selector) const;

    Status Freeze() noexcept;

    AM_NODISCARD size_t size() const noexcept {
        return kernels_.size();
    }

    AM_NODISCARD bool empty() const noexcept {
        return kernels_.empty();
    }

    AM_NODISCARD bool frozen() const noexcept {
        return frozen_;
    }

private:
    bool frozen_ = false;
    std::vector<KernelDescriptor> kernels_{};
};

}// namespace aethermind
#endif
