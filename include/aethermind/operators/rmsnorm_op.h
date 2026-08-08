#ifndef AETHERMIND_OPERATORS_RMS_NORM_OP_H
#define AETHERMIND_OPERATORS_RMS_NORM_OP_H

/// @file rmsnorm_op.h
/// @brief RMS normalization semantics and executable operator declaration.

#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// @brief Dtypes accepted independently for RmsNorm input and weight tensors.
///
/// Input and weight may differ, but each must belong to this set. All RmsNorm
/// validation sites must reference this definition instead of maintaining
/// private copies.
inline const std::array<DataType, 5> kRmsNormSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
        DataType::Float8E4M3FN(),
        DataType::Float8E5M2(),
};

/// @brief Checks whether RmsNorm semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kRmsNormSupportedDTypes`.
inline bool IsRmsNormSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kRmsNormSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// @brief Builds a consistent unsupported-dtype message for RmsNorm.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeRmsNormUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, bfloat16, float8_e4m3fn, and float8_e5m2 dtypes";
    return msg;
}

/// @brief Shape-preserving RMS normalization operator.
///
/// Semantic inference accepts an input of rank at least one and a rank-one
/// weight whose length matches the input's final dimension. Input and weight
/// dtypes are validated independently against `kRmsNormSupportedDTypes`.
/// On `Prepare()`, the operator
/// resolves the backend kernel and stores `eps` as raw bytes in
/// `resolved_kernel_.attrs`; on Run(), dispatches to that kernel via
/// `Operator::InvokeResolvedKernel`. No operator-level workspace is
/// required — scratch (e.g. the per-row RMS accumulator) is owned by the
/// backend kernel itself.
class RmsNormOp final : public Operator {
public:
    using Params = RmsNormParams;

    explicit RmsNormOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kRmsNorm;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "RmsNorm";
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
    // `attrs` carries the raw-byte serialization of params_.eps for the
    // backend kernel to read at execution time.
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif
