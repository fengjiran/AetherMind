#ifndef AETHERMIND_OPERATORS_SILU_OP_H
#define AETHERMIND_OPERATORS_SILU_OP_H

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Single source of truth for the dtype set supported by the Silu operator.
/// All Silu-related validation (semantic analysis in InferSilu, future CPU
/// kernel dispatch) must reference these definitions instead of maintaining
/// private copies. The Phase 1 CPU kernel currently implements only Float32;
/// the semantic layer accepts the full set below.
inline const std::array<DataType, 5> kSiluSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// Returns true if `dtype` is in `kSiluSupportedDTypes`. Used by operator-level
/// validation to keep the dtype check in one place. Backend kernel dispatch
/// must reference this same set when adding new dtype paths.
inline bool IsSiluSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSiluSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for Silu-related
/// validation points. `context` is the caller name (e.g. "Silu",
/// "CpuSiluKernel") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
inline std::string MakeSiluUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// SiLU (Swish) element-wise activation operator.
///
/// Computes `output = input * sigmoid(input) = input / (1 + exp(-input))`.
///
/// Phase 1 scope:
/// - Unary element-wise operation (no broadcasting).
/// - Supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2
///   (see kSiluSupportedDTypes); the CPU kernel currently implements Float32.
/// - Output shape identical to input shape.
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

    AM_NODISCARD const char* Name() const noexcept override {
        return "Silu";
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
