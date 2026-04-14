#ifndef AETHERMIND_BACKEND_BACKEND_H
#define AETHERMIND_BACKEND_BACKEND_H

#include "aethermind/backend/backend_capabilities.h"
#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/kernel_types.h"
#include "macros.h"

namespace aethermind {

class Backend {
public:
    virtual ~Backend() = default;
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;
    AM_NODISCARD virtual const BackendCapabilities& capabilities() const noexcept = 0;

    AM_NODISCARD virtual KernelFn ResolveKernel(const KernelKey& key) const noexcept = 0;

    AM_NODISCARD virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};

}// namespace aethermind
#endif
