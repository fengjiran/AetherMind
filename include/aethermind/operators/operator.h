#ifndef AETHERMIND_OPERATORS_OPERATOR_H
#define AETHERMIND_OPERATORS_OPERATOR_H

#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/runtime/workspace.h"
#include "data_type.h"
#include "macros.h"

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace aethermind {

struct KernelContext;
struct KernelInvocation;
struct WorkspaceBinding;

struct ShapeInfo {
    DataType dtype_{};
    std::vector<int64_t> shape_{};
};

/// Abstract base class for all semantic-layer operators.
///
/// An Operator encapsulates:
/// - Shape inference
/// - Workspace requirement estimation
/// - Kernel resolution and preparation
/// - Runtime execution dispatch
///
/// Lifecycle: Construct → ValidateParams → Prepare → Run (repeated) → Destroy
///
/// Thread safety: Prepare and Run are NOT thread-safe by default.
/// Each invocation should use its own Operator instance or external
/// synchronization.
class Operator {
public:
    virtual ~Operator() = default;

    /// Returns the OpType enum for this operator.
    AM_NODISCARD virtual OpType Type() const noexcept = 0;

    /// Returns a human-readable name for diagnostics and logging.
    /// Default implementation delegates to ToString(type()).
    AM_NODISCARD virtual const char* Name() const noexcept {
        return ToString(Type());
    }

    /// Validates the operator's internal configuration.
    ///
    /// Called after construction and parameter assignment, before Prepare().
    /// Implementations should verify that all required parameters are set
    /// and consistent (e.g., epsilon > 0 for RmsNorm, hidden_size > 0).
    ///
    /// Returns Ok if valid; otherwise returns a Status describing the error.
    AM_NODISCARD virtual Status ValidateParams() const = 0;

    /// Validates that input shapes are compatible with this operator.
    ///
    /// Called during plan building. Checks shape compatibility, dtype
    /// constraints, and dimension consistency — anything computable from
    /// ShapeInfo alone. Data/contiguity/null checks happen in the kernel.
    ///
    /// Returns Ok if compatible; otherwise returns a descriptive error Status.
    AM_NODISCARD virtual Status CheckShapes(std::span<const ShapeInfo> inputs) const = 0;

    /// Infers output shapes from input shapes without executing.
    ///
    /// Used during graph construction and workspace planning.
    ///
    /// \return A vector of ShapeInfo descriptors (one per output),
    ///         or a Status error if inference fails.
    AM_NODISCARD virtual StatusOr<std::vector<ShapeInfo>> InferOutputShapes(
            std::span<const ShapeInfo> inputs) const = 0;

    /// Computes the workspace requirement for this operator.
    ///
    /// Called during execution plan building. The result is stored in the
    /// ExecutionStep and used for unified workspace planning.
    ///
    /// Default returns zero-byte requirement (no scratch space needed).
    ///
    /// \param inputs  Input tensor shapes (for size-dependent estimation).
    AM_NODISCARD virtual WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const ShapeInfo> inputs) const noexcept {
        UNUSED(inputs);
        return {};
    }

    /// Prepares this operator for execution using the provided context.
    ///
    /// Called once during plan building, BEFORE any Run() calls.
    /// Implementations should:
    /// - Resolve the kernel from OperatorContext::backend
    /// - Cache the resolved KernelFunc and attrs
    /// - Validate the resolved kernel matches expectations
    /// - ValidateParams() must have returned Ok before Prepare() is called
    ///
    /// After successful Prepare(), the operator is ready for repeated Run() calls.
    ///
    /// \param ctx  Runtime context providing backend, workspace, and
    ///             kernel selector for resolution.
    /// \return Ok on success; Status error if kernel resolution fails.
    AM_NODISCARD virtual Status Prepare(OperatorContext& ctx) = 0;

    /// Executes this operator on the given inputs and outputs.
    ///
    /// Called once per execution step. Must only be called after
    /// successful Prepare().
    ///
    /// Implementations should:
    /// - Construct a KernelContext using cached kernel metadata
    /// - Invoke the cached KernelFunc with inputs, outputs, and workspace
    ///
    /// \param invocation  Kernel invocation descriptor (op_type + selector).
    /// \param op_ctx      Kernel execution context (device, workspace, etc.).
    /// \param workspace   Pre-bound workspace slice from unified plan.
    /// \return Ok on success; Status error if execution fails.
    AM_NODISCARD virtual Status Run(const KernelInvocation& invocation,
                                    const KernelContext& op_ctx,
                                    const WorkspaceBinding& workspace) const noexcept = 0;

    /// Returns the resolved kernel info for debugging and logging.
    /// Only valid after successful Prepare().
    AM_NODISCARD virtual ResolvedKernel GetResolvedKernel() const noexcept = 0;
};

/// Shared pointer alias for operator lifetime management.
using OperatorPtr = std::shared_ptr<const Operator>;

}// namespace aethermind

#endif
