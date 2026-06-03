#ifndef AETHERMIND_OPERATORS_FUNCTION_OPERATOR_H
#define AETHERMIND_OPERATORS_FUNCTION_OPERATOR_H

#include "aethermind/operators/operator.h"

namespace aethermind {

/// Lightweight Operator adapter that wraps a raw KernelFunc.
///
/// Used when a raw kernel function needs to participate in the Operator
/// execution path (e.g., tests, builder fallback for unregistered operators).
class FunctionOperator final : public Operator {
public:
    FunctionOperator(OpType op_type,
                     KernelFunc fn,
                     std::span<const std::byte> attrs = {},
                     const char* debug_name = nullptr)
        : op_type_(op_type),
          fn_(fn),
          attrs_(attrs.begin(), attrs.end()),
          debug_name_(debug_name) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return op_type_;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return debug_name_ != nullptr ? debug_name_ : "FunctionOperator";
    }

    AM_NODISCARD Status ValidateParams() const override {
        return Status::Ok();
    }

    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec>) const override {
        return Status::Ok();
    }

    AM_NODISCARD StatusOr<std::vector<TensorSpec>> InferOutputShapes(
            std::span<const TensorSpec>) const override {
        return std::vector<TensorSpec>{};
    }

    AM_NODISCARD Status Prepare(OperatorContext&) override {
        return Status::Ok();
    }

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override {
        UNUSED(bindings);
        UNUSED(step_index);
        if (fn_ == nullptr) {
            return Status(StatusCode::kFailedPrecondition, "FunctionOperator kernel function cannot be null");
        }
        return fn_(ctx);
    }

    AM_NODISCARD ResolvedKernel GetResolvedKernel() const noexcept override {
        return ResolvedKernel{
                .op_type = op_type_,
                .fn = fn_,
                .attrs = attrs_,
                .debug_name = debug_name_,
        };
    }

private:
    OpType op_type_;
    KernelFunc fn_;
    std::vector<std::byte> attrs_;
    const char* debug_name_;
};

}// namespace aethermind

#endif
