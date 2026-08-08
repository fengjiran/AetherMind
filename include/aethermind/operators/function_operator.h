#ifndef AETHERMIND_OPERATORS_FUNCTION_OPERATOR_H
#define AETHERMIND_OPERATORS_FUNCTION_OPERATOR_H

/// @file function_operator.h
/// @brief Adapter that exposes a raw kernel function through `Operator`.

#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Lightweight operator adapter around a raw `KernelFunc`.
///
/// Used when a raw kernel function needs to participate in the Operator
/// execution path (e.g., tests, builder fallback for unregistered operators).
class FunctionOperator final : public Operator {
public:
    /// @brief Creates an adapter with immutable kernel metadata.
    ///
    /// @param op_type Operator type reported by the adapter.
    /// @param fn Non-owning kernel function pointer invoked by `Run()`.
    /// @param attrs Attributes copied into the resolved-kernel record.
    /// @param debug_name Optional static diagnostic name; not owned.
    FunctionOperator(OpType op_type,
                     KernelFunc fn,
                     std::span<const std::byte> attrs = {},
                     const char* debug_name = nullptr)
        : resolved_kernel_{
                  .op_type = op_type,
                  .fn = fn,
                  .attrs = std::vector<std::byte>(attrs.begin(), attrs.end()),
                  .debug_name = debug_name,
          } {}

    AM_NODISCARD OpType Type() const noexcept override {
        return resolved_kernel_.op_type;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return resolved_kernel_.debug_name != nullptr ? resolved_kernel_.debug_name : "FunctionOperator";
    }

    AM_NODISCARD Status Prepare(OperatorContext&) override {
        return Status::Ok();
    }

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override {
        UNUSED(bindings);
        UNUSED(step_index);
        if (resolved_kernel_.fn == nullptr) {
            return Status::FailedPrecondition("FunctionOperator kernel function cannot be null");
        }
        return resolved_kernel_.fn(ctx);
    }

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
