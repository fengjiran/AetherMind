#ifndef AETHERMIND_OPERATORS_OPS_SILU_MUL_OP_H
#define AETHERMIND_OPERATORS_OPS_SILU_MUL_OP_H

/// @file silu_mul_op.h
/// @brief Fused SiLU-multiply semantics and executable operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Dtypes accepted by both SiluMul inputs.
///
/// Both inputs must use the same dtype from this set; no implicit conversion
/// policy is declared.
inline const std::array<DataType, 5> kSiluMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// @brief Checks whether SiluMul semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kSiluMulSupportedDTypes`.
inline bool IsSiluMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSiluMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for SiluMul.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeSiluMulUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// @brief Fused SiLU-multiplication operator.
///
/// Computes `output = silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`.
/// Inputs `gate` and `up` are broadcast according to NumPy semantics.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class SiluMulOp final : public Operator {
public:
    using Params = SiluMulParams;

    explicit SiluMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kSiluMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "SiluMul";
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
