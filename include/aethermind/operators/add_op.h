// Semantic-layer operator for elementwise addition with NumPy-style broadcasting.
//
// Inputs (per CheckInputSpecs / InferOutputShapes):
//   - inputs[0]: lhs tensor, any rank
//   - inputs[1]: rhs tensor, any rank (must be broadcast-compatible with lhs)
// Both inputs must share a dtype from kAddSupportedDTypes
// (float32, float64, bfloat16, int32, int64). The output shape is the
// broadcast of the two input shapes; deferred broadcastability checks
// that depend on runtime shapes are emitted via InferenceResult::runtime_checks.
//
// Lifecycle and thread-safety follow the Operator base class contract:
// Construct -> ValidateParams() -> Prepare() -> Run() (repeated).
// Instances are not thread-safe; concurrent Run() requires external
// synchronization or one instance per thread.

#ifndef AETHERMIND_OPERATORS_ADD_OP_H
#define AETHERMIND_OPERATORS_ADD_OP_H

#include "aethermind/dtypes/data_type.h"
#include "aethermind/model/graph/op_params.h"
#include "aethermind/operators/operator.h"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace aethermind {

/// Single source of truth for the dtype set supported by the Add operator.
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

/// Returns true if `dtype` is in `kAddSupportedDTypes`. Used by operator-level
/// validation and backend kernel dispatch to keep the dtype check in one place.
inline bool IsAddSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kAddSupportedDTypes, [&](const DataType& supported) {
        return dtype == supported;
    });
}

/// Builds a consistent "unsupported dtype" error message for Add-related
/// validation points. `context` is the caller name (e.g. "Add", "CpuAddKernel",
/// "LaunchAdd") prepended to a fixed list of supported dtypes, so every
/// validation site reports the same set.
inline std::string MakeAddUnsupportedDTypeMessage(std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float64, bfloat16, int32, and int64 tensors";
    return msg;
}

/// Elementwise Add operator with NumPy-style broadcasting.
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

    AM_NODISCARD Status ValidateParams() const override;
    AM_NODISCARD Status CheckInputSpecs(std::span<const TensorSpec> inputs) const override;
    AM_NODISCARD StatusOr<InferenceResult> InferOutputShapes(
            std::span<const TensorSpec> inputs) const override;

    AM_NODISCARD Status Prepare(OperatorContext& ctx) override;

    AM_NODISCARD Status Run(KernelContext& ctx,
                            const RuntimeBindingContext& bindings,
                            size_t step_index) const noexcept override;

    AM_NODISCARD const ResolvedKernel& GetResolvedKernel() const noexcept override {
        return resolved_kernel_;
    }

private:
    Params params_{};
    // Written only by Prepare(); read by Run() and GetResolvedKernel().
    ResolvedKernel resolved_kernel_{};
};

}// namespace aethermind

#endif