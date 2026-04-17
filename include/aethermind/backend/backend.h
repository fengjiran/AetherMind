#ifndef AETHERMIND_BACKEND_BACKEND_H
#define AETHERMIND_BACKEND_BACKEND_H

#include "aethermind/backend/backend_capabilities.h"
#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/backend/resolved_kernel.h"
#include "macros.h"

namespace aethermind {

class Backend {
public:
    virtual ~Backend() = default;
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;
    AM_NODISCARD virtual const BackendCapabilities& capabilities() const noexcept = 0;

    AM_NODISCARD virtual KernelFunc ResolveKernel(
            OpType op_type,
            const KernelSelector& selector) const noexcept = 0;

    AM_NODISCARD virtual StatusOr<ResolvedKernel> ResolveKernelInfo(
            OpType op_type,
            const KernelSelector& selector) const noexcept = 0;

    AM_NODISCARD virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};

}// namespace aethermind
#endif
