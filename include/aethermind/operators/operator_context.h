#ifndef AETHERMIND_OPERATORS_OPERATOR_CONTEXT_H
#define AETHERMIND_OPERATORS_OPERATOR_CONTEXT_H

#include "aethermind/backend/backend_fwd.h"
#include "aethermind/backend/kernel_selector.h"
#include "macros.h"

namespace aethermind {

class WorkspaceArena;

/// Runtime dependencies passed into semantic-layer Operator::Run calls.
struct OperatorContext {
    /// Backend used to resolve and invoke kernels for the operator.
    Backend* backend = nullptr;

    /// Optional registry view for diagnostics and tests; ownership stays with the backend.
    const KernelRegistry* kernel_registry = nullptr;

    /// Runtime workspace arena used by operators that require scratch space.
    WorkspaceArena* workspace = nullptr;

    /// Kernel capability request used when resolving backend kernels.
    KernelSelector selector{};

    bool enable_profiling = false;
    bool enable_debug_check = false;

    AM_NODISCARD ExecPhase phase() const noexcept {
        return selector.phase;
    }
};

}// namespace aethermind

#endif
