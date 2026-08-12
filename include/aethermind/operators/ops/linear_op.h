#ifndef AETHERMIND_OPERATORS_OPS_LINEAR_OP_H
#define AETHERMIND_OPERATORS_OPS_LINEAR_OP_H

/// @file linear_op.h
/// @brief Linear-transformation semantics and executable operator declaration.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

#include <array>
#include <string>

namespace aethermind {

/// @brief Dtypes accepted for a Linear activation input.
///
/// All Linear activation validation sites must reference this set instead of
/// maintaining private copies.
inline const std::array<DataType, 5> kLinearSupportedActivationDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// @brief Checks whether Linear accepts an activation dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kLinearSupportedActivationDTypes`.
inline bool IsLinearSupportedActivationDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kLinearSupportedActivationDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Dtypes accepted for a Linear weight input.
///
/// Quantized int8 and int4 weights are valid semantic inputs. Output dtype
/// follows the activation dtype.
inline const std::array<DataType, 7> kLinearSupportedWeightDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
        DataType::Int(8),
        DataType::Int(4),
};

/// @brief Checks whether Linear accepts a weight dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kLinearSupportedWeightDTypes`.
inline bool IsLinearSupportedWeightDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kLinearSupportedWeightDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-activation-dtype message for Linear.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted activation dtypes.
inline std::string MakeLinearUnsupportedActivationDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " activation only supports float32, float16, bfloat16, "
           "float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// @brief Builds a consistent unsupported-weight-dtype message for Linear.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted weight dtypes.
inline std::string MakeLinearUnsupportedWeightDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " weight only supports float32, float16, bfloat16, "
           "float8_e4m3fn, float8_e5m2, int8, and int4 dtypes";
    return msg;
}


/// @brief Semantic operator for `output = input @ weight.T`.
///
/// Weight shape convention: [out_features, in_features] (PyTorch/HF row-major).
/// Input shape: [..., in_features] with rank at least one.
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

#endif// AETHERMIND_OPERATORS_OPS_LINEAR_OP_H
