//
// Created by richard on 4/15/26.
//

#ifndef AETHERMIND_BACKEND_KERNEL_REGISTRY_H
#define AETHERMIND_BACKEND_KERNEL_REGISTRY_H

#include "aethermind/backend/kernel_key.h"
#include "aethermind/backend/kernel_types.h"
#include "macros.h"

#include <unordered_map>

namespace aethermind {

class KernelRegistry {
public:
    void Register(const KernelKey& key, KernelFn fn);
    AM_NODISCARD KernelFn Find(const KernelKey& key) const noexcept;

private:
    std::unordered_map<KernelKey, KernelFn> kernels_;
};

}// namespace aethermind
#endif
