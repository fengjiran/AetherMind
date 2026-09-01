#ifndef AETHERMIND_BACKEND_BACKEND_H
#define AETHERMIND_BACKEND_BACKEND_H

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/backend/kernel_types.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/macros.h"
#include "aethermind/operators/op_params.h"

namespace aethermind {

class Backend {
public:
    virtual ~Backend() = default;
    AM_NODISCARD virtual DeviceType device_type() const noexcept = 0;

    /// Resolves a kernel and freezes metadata derived from typed semantic
    /// parameters into the returned value, including its scratch-space
    /// requirement.
    ///
    /// This is the sole planning-time kernel-resolution entry point. The
    /// returned ResolvedKernel is later owned by an ExecutionStep and invoked
    /// without another backend lookup.
    AM_NODISCARD virtual StatusOr<ResolvedKernel> PrepareKernel(
            OpType op_type,
            const KernelSelector& selector,
            const OpParams& params) const = 0;

    AM_NODISCARD virtual const KernelRegistry* TryGetKernelRegistryForDebug() const noexcept = 0;
};

} // namespace aethermind
#endif
