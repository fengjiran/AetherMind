#ifndef AETHERMIND_OPERATORS_ADD_OP_H
#define AETHERMIND_OPERATORS_ADD_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

class AddOp final : public Operator {
public:
    using Params = AddParams;

    explicit AddOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kAdd;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Add";
    }

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override;
    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override;

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override;

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
