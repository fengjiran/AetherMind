#ifndef AETHERMIND_OPERATORS_OPERATOR_H
#define AETHERMIND_OPERATORS_OPERATOR_H

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/operators/op_type.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/runtime/workspace.h"
#include "aethermind/shape_inference/shape_constraint.h"
#include "aethermind/shape_inference/shape_symbol.h"
#include "aethermind/operators/inference_result.h"
#include "aethermind/shape_inference/tensor_spec.h"
#include "macros.h"

#include <cstddef>
#include <memory>
#include <span>
#include <vector>

namespace aethermind {

struct KernelContext;
class RuntimeBindingContext;

/// Abstract base class for all semantic-layer operators.
///
/// Encapsulates param validation, shape inference, workspace estimation,
/// kernel resolution, and runtime execution behind a uniform contract.
///
/// Lifecycle: Construct → ValidateParams() → Prepare() → Run() (repeated)
/// → destruction. `Prepare()` resolves and caches the kernel; `Run()` may
/// then be invoked multiple times. The destructor releases cached state
/// and must not run concurrently with an in-flight `Run()`.
///
/// Invariant: `Prepare()` must return Ok before `Run()` or
/// `GetResolvedKernel()` may be called. `resolved_kernel_` is written only
/// by `Prepare()` and read by `Run()` / `GetResolvedKernel()`.
///
/// Thread safety: instances are not thread-safe. Concurrent `Run()` calls
/// on the same instance require external synchronization, or use one
/// instance per thread.
class Operator {
public:
    virtual ~Operator() = default;

    /// Returns the OpType enum for this operator.
    AM_NODISCARD virtual OpType Type() const noexcept = 0;

    /// Returns a human-readable name for diagnostics and logging.
    /// Default implementation delegates to `ToString(Type())`.
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

    /// Validates that input TensorSpecs are compatible with this operator.
    ///
    /// Called during plan building. Checks dtype constraints,
    /// rank and dimension consistency — anything computable from
    /// TensorSpec alone. Data/contiguity/null checks happen in the kernel.
    ///
    /// Returns Ok if compatible; otherwise returns a descriptive error Status.
    AM_NODISCARD virtual Status CheckInputSpecs(std::span<const TensorSpec> inputs) const = 0;

    /// Infers output shapes from input shapes without executing.
    ///
    /// Used during graph construction and workspace planning. May emit
    /// deferred `ShapeConstraint`s in `InferenceResult::runtime_checks` for
    /// conditions that cannot be proven until concrete runtime shapes are
    /// known.
    ///
    /// \return Inferred output specs (and any deferred runtime checks),
    ///         or a Status error if inference fails.
    AM_NODISCARD virtual StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const = 0;

    /// Computes the workspace requirement for this operator.
    ///
    /// Called during execution plan building. The result is stored in the
    /// ExecutionStep and used for unified workspace planning.
    ///
    /// Default returns zero-byte requirement (no scratch space needed).
    ///
    /// \param inputs  Input tensor shapes (for size-dependent estimation).
    AM_NODISCARD virtual WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept {
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
    /// successful Prepare(). Declared `noexcept`: implementations report
    /// failures only through the returned `Status`, never by throwing.
    ///
    /// Implementations should:
    /// - Construct per-call kernel params from runtime bindings when needed
    /// - Invoke the cached KernelFunc with inputs, outputs, and workspace
    ///
    /// \param ctx          Mutable kernel context. Operators may set
    ///                     ctx.kernel_params before invoking the kernel.
    /// \param bindings     Per-step runtime tensor bindings.
    /// \param step_index   Index of the current execution step in the plan,
    ///                     used to retrieve per-step tensor bindings.
    /// \return Ok on success; Status error if execution fails.
    AM_NODISCARD virtual Status Run(KernelContext& ctx,
                                    const RuntimeBindingContext& bindings,
                                    size_t step_index) const noexcept = 0;

    /// Returns the resolved kernel info for execution, debugging, and logging.
    /// Only valid after successful Prepare().
    AM_NODISCARD virtual const ResolvedKernel& GetResolvedKernel() const noexcept = 0;

protected:
    /// Shared kernel invocation path for subclasses.
    ///
    /// If `ResolvedKernel::params_builder` is set, stack-allocates an
    /// aligned buffer of `kMaxKernelParamsSize` bytes, asks the builder to
    /// placement-construct the backend-specific params struct into it,
    /// points `ctx.kernel_params` at the buffer, then invokes the kernel
    /// function. If `params_builder` is null, invokes the kernel directly
    /// with whatever `ctx.kernel_params` the caller has already set.
    ///
    /// Lifetime: the stack buffer is local to this call. The kernel must
    /// not retain pointers into it past return.
    ///
    /// Performance: stack allocation avoids per-call heap traffic on the
    /// execution hot path. `kMaxKernelParamsSize` bounds the cost.
    AM_NODISCARD Status InvokeResolvedKernel(KernelContext& ctx,
                                             std::span<const TensorView> inputs,
                                             std::span<const MutableTensorView> outputs) const noexcept {
        const ResolvedKernel& resolved = GetResolvedKernel();
        // Declared at function scope so the buffer outlives the kernel call;
        // previously it was scoped inside the if block, leaving
        // ctx.kernel_params dangling when resolved.fn(ctx) ran below.
        alignas(std::max_align_t) std::byte buffer[kMaxKernelParamsSize];
        if (resolved.params_builder != nullptr) {
            AM_RETURN_IF_ERROR(resolved.params_builder(inputs, outputs, buffer));
            ctx.kernel_params = buffer;
        }
        return resolved.fn(ctx);
    }
};

/// Shared pointer alias for operator lifetime management.
using OperatorPtr = std::shared_ptr<const Operator>;

}// namespace aethermind

#endif
