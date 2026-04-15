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
    Status Resolve(OpType op_type,
                   const KernelSelector& selector,
                   const KernelDescriptor** out) const noexcept;

private:
    std::vector<KernelDescriptor> kernels_{};
};

}// namespace aethermind
#endif
