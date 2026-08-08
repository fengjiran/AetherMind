#ifndef AETHERMIND_OPERATORS_OPERATOR_H
#define AETHERMIND_OPERATORS_OPERATOR_H

/// @file operator.h
/// @brief Runtime interface shared by executable operator implementations.

#include "aethermind/backend/kernel_context.h"
#include "aethermind/backend/resolved_kernel.h"
#include "aethermind/operators/operator_context.h"
#include "aethermind/shape_inference/tensor_spec.h"

namespace aethermind {

struct KernelContext;
class RuntimeBindingContext;

/// @brief Abstract base class for executable semantic operators.
///
/// Encapsulates kernel resolution and runtime execution behind a uniform
/// contract. Parameter validation and shape inference are performed via the
/// free function InferOperator in operator_inference.h, not through virtual
/// methods on this class.
///
/// Lifecycle: construct, call `Prepare()` once, call `Run()` repeatedly, then
/// destroy after all in-flight calls have completed.
/// `Prepare()` resolves and caches the kernel; `Run()` may
/// then be invoked multiple times. The destructor releases cached state
/// and must not run concurrently with an in-flight `Run()`.
///
/// @pre `Prepare()` must return Ok before `Run()` or
/// `GetResolvedKernel()` may be called. `resolved_kernel_` is written only
/// by `Prepare()` and read by `Run()` / `GetResolvedKernel()`.
///
/// @note Instances are not thread-safe. Concurrent calls on one instance
///       require external synchronization.
class Operator {
public:
    virtual ~Operator() = default;

    /// @brief Returns the type used for operator and kernel dispatch.
    ///
    /// @return Operator type associated with this instance.
    AM_NODISCARD virtual OpType Type() const noexcept = 0;

    /// @brief Returns a human-readable name for diagnostics and logging.
    ///
    /// @return Null-terminated name whose lifetime is static.
    /// @note The default implementation delegates to `ToString(Type())`.
    AM_NODISCARD virtual const char* Name() const noexcept {
        return ToString(Type());
    }

    /// @brief Computes the scratch-workspace requirement for this operator.
    ///
    /// Called during execution plan building. The result is stored in the
    /// ExecutionStep and used for unified workspace planning.
    ///
    /// The default implementation returns a zero-byte requirement.
    ///
    /// @param inputs Input tensor specifications used for size-dependent estimates.
    /// @return Workspace size and alignment required by one invocation.
    AM_NODISCARD virtual WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept {
        UNUSED(inputs);
        return {};
    }

    /// @brief Resolves and caches the kernel used by subsequent invocations.
    ///
    /// Called once during plan building, BEFORE any Run() calls.
    /// @param ctx Runtime dependencies and kernel-selection criteria.
    /// @return Ok on success, or an error status when kernel resolution fails.
    /// @pre Semantic inference for this operator has succeeded.
    /// @post On success, `Run()` and `GetResolvedKernel()` may be called.
    virtual Status Prepare(OperatorContext& ctx) = 0;

    /// @brief Executes one prepared operator step using runtime tensor bindings.
    ///
    /// Called once per execution step. Must only be called after
    /// successful Prepare(). Declared `noexcept`: implementations report
    /// failures only through the returned `Status`, never by throwing.
    ///
    /// @param ctx Mutable context passed to the resolved kernel.
    /// @param bindings Runtime tensor bindings for all execution steps.
    /// @param step_index Step whose input and output bindings are used.
    /// @return Ok on success, or an error status for binding or kernel failure.
    /// @pre `Prepare()` has completed successfully.
    virtual Status Run(KernelContext& ctx,
                       const RuntimeBindingContext& bindings,
                       size_t step_index) const noexcept = 0;

    /// @brief Returns the kernel metadata cached by `Prepare()`.
    ///
    /// @return Borrowed reference valid for the lifetime of this operator.
    /// @pre `Prepare()` has completed successfully.
    AM_NODISCARD virtual const ResolvedKernel& GetResolvedKernel() const noexcept = 0;

protected:
    /// @brief Invokes the resolved kernel with optional per-call parameters.
    ///
    /// If `ResolvedKernel::params_builder` is set, stack-allocates an
    /// aligned buffer of `kMaxKernelParamsSize` bytes, asks the builder to
    /// placement-construct the backend-specific params struct into it,
    /// points `ctx.kernel_params` at the buffer, then invokes the kernel
    /// function. If `params_builder` is null, invokes the kernel directly
    /// with whatever `ctx.kernel_params` the caller has already set.
    ///
    /// @param ctx Mutable context passed to the resolved kernel.
    /// @param inputs Borrowed input tensor views valid for this call.
    /// @param outputs Borrowed output tensor views valid for this call.
    /// @return Status returned by parameter construction or kernel execution.
    /// @pre `GetResolvedKernel()` returns a non-null kernel function.
    /// @note Parameter storage is stack-backed to avoid heap traffic. Kernels
    ///       must not retain `ctx.kernel_params` after returning.
    AM_NODISCARD Status InvokeResolvedKernel(KernelContext& ctx,
                                             std::span<const TensorView> inputs,
                                             std::span<const MutableTensorView> outputs) const noexcept {
        const ResolvedKernel& resolved = GetResolvedKernel();
        // Keep parameter storage alive through the kernel call; the builder
        // publishes a borrowed pointer to this buffer through `ctx`.
        alignas(std::max_align_t) std::byte buffer[kMaxKernelParamsSize];
        if (resolved.params_builder != nullptr) {
            AM_RETURN_IF_ERROR(resolved.params_builder(inputs, outputs, buffer));
            ctx.kernel_params = buffer;
        }
        return resolved.fn(ctx);
    }
};

/// @brief Shared immutable ownership of an operator instance.
using OperatorPtr = std::shared_ptr<const Operator>;

}// namespace aethermind

#endif
