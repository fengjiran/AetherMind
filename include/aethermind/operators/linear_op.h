#ifndef AETHERMIND_OPERATORS_LINEAR_OP_H
#define AETHERMIND_OPERATORS_LINEAR_OP_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

#include <array>
#include <string>

namespace aethermind {

/// Single source of truth for the dtype set supported by the Linear operator's
/// activation input. All Linear-related validation (semantic analysis in
/// InferLinear, future CPU kernel dispatch) must reference these definitions
/// instead of maintaining private copies. The semantic layer accepts these
/// dtypes; the Phase 1 CPU kernel currently implements only Float32.
inline const std::array<DataType, 5> kLinearSupportedActivationDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// Returns true if `dtype` is a valid Linear activation dtype
/// (float32, float16, bfloat16, float8_e4m3fn, float8_e5m2). Backend kernel
/// dispatch must reference this same set when adding new dtype paths.
inline bool IsLinearSupportedActivationDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kLinearSupportedActivationDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Single source of truth for the dtype set supported by the Linear operator's
/// weight input. The semantic layer accepts these dtypes (including quantized
/// int8/int4); the Phase 1 CPU kernel currently implements only Float32.
/// Output dtype follows activation dtype.
inline const std::array<DataType, 7> kLinearSupportedWeightDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
        DataType::Int(8),
        DataType::Int(4),
};

/// Returns true if `dtype` is a valid Linear weight dtype
/// (float32, float16, bfloat16, float8_e4m3fn, float8_e5m2, int8, int4).
/// Backend kernel dispatch must reference this same set when adding new dtype
/// paths.
inline bool IsLinearSupportedWeightDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kLinearSupportedWeightDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported activation dtype" error message for
/// Linear-related validation points. `context` is the caller name
/// (e.g. "Linear", "CpuLinearKernel") prepended to a fixed list of supported
/// activation dtypes, so every validation site reports the same set.
inline std::string MakeLinearUnsupportedActivationDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " activation only supports float32, float16, bfloat16, "
           "float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// Builds a consistent "unsupported weight dtype" error message for
/// Linear-related validation points. `context` is the caller name
/// (e.g. "Linear", "CpuLinearKernel") prepended to a fixed list of supported
/// weight dtypes, so every validation site reports the same set.
inline std::string MakeLinearUnsupportedWeightDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " weight only supports float32, float16, bfloat16, "
           "float8_e4m3fn, float8_e5m2, int8, and int4 dtypes";
    return msg;
}


/// Semantic operator for linear transformation: output = input @ weight.T
///
/// Weight shape convention: [out_features, in_features] (PyTorch/HF row-major).
/// Input shape: [..., in_features] (rank 1 or 2 in Phase 1).
/// Output shape: [..., out_features] (same rank as input).
class LinearOp final : public Operator {
public:
    using Params = LinearParams;

    explicit LinearOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kLinear;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "Linear";
    }

    AM_NODISCARD WorkspaceRequirement ComputeWorkspaceRequirement(
            std::span<const TensorSpec> inputs) const noexcept override {
        UNUSED(inputs);
        return {};
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

#endif// AETHERMIND_OPERATORS_LINEAR_OP_H
