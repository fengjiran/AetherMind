#ifndef AETHERMIND_OPERATORS_ELEMENTWISE_MUL_OP_H
#define AETHERMIND_OPERATORS_ELEMENTWISE_MUL_OP_H

/// @file elementwise_mul_op.h
/// @brief Elementwise multiplication semantics and operator declaration.

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Dtypes accepted by ElementwiseMul semantic inference.
///
/// All ElementwiseMul validation sites must reference this set instead of
/// maintaining private copies.
inline const std::array<DataType, 3> kElementwiseMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// @brief Checks whether ElementwiseMul semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kElementwiseMulSupportedDTypes`.
inline bool IsElementwiseMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kElementwiseMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for ElementwiseMul.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeElementwiseMulUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

/// @brief Elementwise multiplication operator with NumPy-style broadcasting.
class ElementwiseMulOp final : public Operator {
public:
    using Params = ElementwiseMulParams;

    explicit ElementwiseMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kElementwiseMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "ElementwiseMul";
    }

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
