#ifndef AETHERMIND_OPERATORS_SILU_MUL_OP_H
#define AETHERMIND_OPERATORS_SILU_MUL_OP_H

#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Single source of truth for the dtype set supported by the SiluMul operator.
/// All SiluMul-related validation (semantic analysis in InferSiluMul, future
/// CPU kernel dispatch) must reference these definitions instead of maintaining
/// private copies. The semantic layer accepts mixed-precision (lhs/gate and
/// rhs/up may differ, each independently drawn from this set); the Phase 1 CPU
/// kernel currently implements only Float32.
inline const std::array<DataType, 5> kSiluMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// Returns true if `dtype` is in `kSiluMulSupportedDTypes`. Used by
/// operator-level validation to keep the dtype check in one place. Backend
/// kernel dispatch must reference this same set when adding new dtype paths.
inline bool IsSiluMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kSiluMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for SiluMul-related
/// validation points. `context` is the caller name (e.g. "SiluMul lhs",
/// "CpuSiluMulKernel") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
inline std::string MakeSiluMulUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// Fused SiLU-multiplication operator.
///
/// Computes `output = silu(gate) * up` where `silu(x) = x / (1 + exp(-x))`.
/// Inputs `gate` and `up` are broadcast according to NumPy semantics.
///
/// Phase 1 scope:
/// - Binary element-wise operation with broadcasting.
/// - Supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2
///   (see kSiluMulSupportedDTypes); the CPU kernel currently implements Float32.
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
