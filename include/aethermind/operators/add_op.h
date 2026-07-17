#ifndef AETHERMIND_OPERATORS_ADD_OP_H
#define AETHERMIND_OPERATORS_ADD_OP_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

// Single source of truth for the dtype set supported by the Add operator.
// All Add-related validation (op params, CPU kernel dispatch, constant folding)
// must reference these definitions instead of maintaining private copies.
inline const std::array<DataType, 5> kAddSupportedDTypes = {
        DataType::Float32(),
        DataType::Double(),
        DataType::BFloat(16),
        DataType::Int(32),
        DataType::Int(64),
};

inline bool IsAddSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kAddSupportedDTypes, [&](const DataType& supported) {
        return dtype == supported;
    });
}

inline std::string MakeAddUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float64, bfloat16, int32, and int64 tensors";
    return msg;
}

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
