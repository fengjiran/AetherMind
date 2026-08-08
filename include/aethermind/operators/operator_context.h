#ifndef AETHERMIND_OPERATORS_OPERATOR_CONTEXT_H
#define AETHERMIND_OPERATORS_OPERATOR_CONTEXT_H

/// @file operator_context.h
/// @brief Borrowed runtime dependencies used while preparing operators.

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/kernel_selector.h"
#include "aethermind/base/macros.h"

namespace aethermind {

class WorkspaceArena;

/// @brief Runtime dependencies used to resolve executable operator kernels.
///
/// All pointer members are non-owning. Their pointees must outlive every
/// `Prepare()` call that receives this context.
struct OperatorContext {
    /// @brief Backend used to resolve kernels.
    Backend* backend = nullptr;

    /// @brief Optional registry view for diagnostics and tests.
    ///
    /// Ownership remains with the backend.
    const KernelRegistry* kernel_registry = nullptr;

    /// @brief Optional workspace arena borrowed by operators requiring scratch space.
    WorkspaceArena* workspace = nullptr;

    /// @brief Capability request used during kernel resolution.
    KernelSelector selector{};

    bool enable_profiling = false;
    bool enable_debug_check = false;

    /// @brief Returns the execution phase encoded by the kernel selector.
    ///
    /// @return Current prefill or decode phase.
    AM_NODISCARD ExecPhase phase() const noexcept {
        return selector.phase;
    }
};

}// namespace aethermind

#endif
