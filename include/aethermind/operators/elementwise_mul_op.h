#ifndef AETHERMIND_OPERATORS_ELEMENTWISE_MUL_OP_H
#define AETHERMIND_OPERATORS_ELEMENTWISE_MUL_OP_H

#include <algorithm>
#include <array>
#include <ranges>
#include <string>
#include <string_view>

#include "aethermind/dtypes/data_type.h"
#include "aethermind/operators/op_params.h"
#include "aethermind/operators/operator.h"

namespace aethermind {

/// Single source of truth for the dtype set supported by the ElementwiseMul
/// operator. All ElementwiseMul-related validation (semantic analysis in
/// InferElementwiseMul, CPU kernel dispatch) must reference these definitions
/// instead of maintaining private copies. The Phase 1 CPU kernel currently
/// implements only Float32 and BFloat16; the semantic layer accepts the full
/// set below.
inline const std::array<DataType, 3> kElementwiseMulSupportedDTypes = {
        DataType::Float32(),
        DataType::Float(16),
        DataType::BFloat(16),
};

/// Returns true if `dtype` is in `kElementwiseMulSupportedDTypes`. Used by
/// operator-level validation to keep the dtype check in one place. Backend
/// kernel dispatch must reference this same set when adding new dtype paths.
inline bool IsElementwiseMulSupportedDType(const DataType& dtype) noexcept {
    return std::ranges::any_of(kElementwiseMulSupportedDTypes,
                               [&](const DataType& supported) {
                                   return dtype == supported;
                               });
}

/// Builds a consistent "unsupported dtype" error message for
/// ElementwiseMul-related validation points. `context` is the caller name
/// (e.g. "ElementwiseMul lhs", "CpuElementwiseMulKernel rhs") prepended to a
/// fixed list of supported dtypes, so every validation site reports the same
/// set.
inline std::string MakeElementwiseMulUnsupportedDTypeMessage(
        std::string_view context) {
    std::string msg{context};
    msg += " only supports float32, float16, and bfloat16 dtypes";
    return msg;
}

class ElementwiseMulOp final : public Operator {
public:
    using Params = ElementwiseMulParams;

    explicit ElementwiseMulOp(Params params) noexcept : params_(params) {}

    AM_NODISCARD OpType Type() const noexcept override {
        return OpType::kElementwiseMul;
    }

    AM_NODISCARD const char* Name() const noexcept override {
        return "ElementwiseMul";
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
