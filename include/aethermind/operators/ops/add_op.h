#ifndef AETHERMIND_OPERATORS_OPS_ADD_OP_H
#define AETHERMIND_OPERATORS_OPS_ADD_OP_H

/// @file add_op.h
/// @brief Elementwise addition semantics and executable operator declaration.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

#include <algorithm>
#include <array>
#include <string_view>

namespace aethermind {

/// @brief Dtypes accepted by Add semantic inference.
///
/// All Add-related validation (op params, CPU kernel dispatch, constant
/// folding) must reference these definitions instead of maintaining private
/// copies. Add kernel registrations (see add_entry.cpp) statically assert
/// that their entry count matches kAddSupportedDTypes.size().
inline const std::array<DataType, 5> kAddSupportedDTypes = {
        DataType::Float32(),
        DataType::Double(),
        DataType::BFloat(16),
        DataType::Int(32),
        DataType::Int(64),
};

/// @brief Checks whether Add semantic inference accepts a dtype.
///
/// @param dtype Data type to check.
/// @return True if `dtype` is in `kAddSupportedDTypes`.
inline bool IsAddSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kAddSupportedDTypes, [&](const DataType& supported) {
        return dtype == supported;
    });
}

/// @brief Builds a consistent unsupported-dtype message for Add.
///
/// validation points. `context` is the caller name (e.g. "Add", "CpuAddKernel",
/// "LaunchAdd") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
///
/// @param context Caller name prepended to the supported-dtype description.
/// @return Error message containing `context` and the accepted dtype set.
inline std::string MakeAddUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float64, bfloat16, int32, and int64 tensors";
    return msg;
}

/// @brief Elementwise Add operator with NumPy-style broadcasting.
///
/// Validates that both inputs share a dtype from `kAddSupportedDTypes` and
/// are broadcast-compatible; infers the broadcast output shape. On Prepare(),
/// resolves the backend kernel; on Run(), dispatches to it via
/// `Operator::InvokeResolvedKernel`.
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
