#ifndef AETHERMIND_OPERATORS_MATMUL_OP_H
#define AETHERMIND_OPERATORS_MATMUL_OP_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Single source of truth for the dtype set supported by the MatMul
/// operator's lhs and rhs inputs. All MatMul-related validation (semantic
/// analysis in InferMatMul, future CPU kernel dispatch) must reference these
/// definitions instead of maintaining private copies. The semantic layer
/// accepts these dtypes; the Phase 1 CPU kernel currently implements only
/// Float32. Output dtype follows lhs dtype.
inline const std::array<DataType, 6> kMatMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
        DataType::Int(8),
};

/// Returns true if `dtype` is a valid MatMul dtype
/// (float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, int8). Backend
/// kernel dispatch must reference this same set when adding new dtype paths.
inline bool IsMatMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kMatMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for MatMul-related
/// validation points. `context` is the caller name (e.g. "MatMul",
/// "CpuMatMulKernel") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
inline std::string MakeMatMulUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, "
           "float8_e4m3fn, float8_e5m2, and int8 dtypes";
    return msg;
}

/// Semantic operator for batched matrix multiplication.
///
/// Computes `output = lhs @ effective_rhs` where:
/// - `effective_rhs = rhs` when `transpose_rhs == false` (rhs shape [..., K, N])
/// - `effective_rhs = rhs^T` when `transpose_rhs == true`  (rhs shape [..., N, K])
///
/// Both inputs must have rank >= 2. Trailing two axes are the matrix axes;
/// leading axes are batch axes broadcast according to NumPy semantics.
/// Output shape: `broadcast(lhs_batch, rhs_batch) + [M, N]` where
/// `M = lhs.shape[-2]` and `N = effective_rhs.shape[-1]`.
///
/// Phase 1 scope:
/// - Semantic layer accepts float32, float16, bfloat16, float8_e4m3fn,
///   float8_e5m2, and int8 dtypes (output dtype follows lhs dtype).
/// - CPU kernel currently implements float32 only.
/// - Supports transpose_rhs for PyTorch/HF-style weight storage.
///
/// Kernel execution is not yet wired; Run() returns Unimplemented after
/// binding validation. A CPU kernel can be added later without changing the
/// semantic-layer contract.
class MatMulOp final : public Operator {
public:
    using Params = MatMulParams;

    explicit MatMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kMatMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "MatMul";
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

#endif// AETHERMIND_OPERATORS_MATMUL_OP_H
