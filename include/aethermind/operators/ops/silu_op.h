#ifndef AETHERMIND_OPERATORS_OPS_SILU_OP_H
#define AETHERMIND_OPERATORS_OPS_SILU_OP_H

/// @file silu_op.h
/// @brief SiLU activation semantics and executable operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Dtypes accepted by SiLU semantic inference.
///
/// All SiLU validation sites must reference this set instead of maintaining
/// private copies.
inline const std::array<DataType, 5> kSiluSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// @brief Checks whether SiLU semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kSiluSupportedDTypes`.
inline bool IsSiluSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSiluSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for SiLU.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeSiluUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// @brief SiLU (Swish) elementwise activation operator.
///
/// Computes `output = input * sigmoid(input) = input / (1 + exp(-input))`.
///
/// This is a unary operation with no broadcasting; the output tensor spec is
/// identical to the input tensor spec.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class SiluOp final : public Operator {
public:
    using Params = SiluParams;

    explicit SiluOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kSilu;
    }

    Status Prepare(OperatorContext& ctx) override;

    Status Run(KernelContext& ctx,
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
