#include "aethermind/operators/op_type.h"
#include "const_eval_internal.h"

#include <algorithm>

namespace aethermind {
namespace {

struct EvaluatorEntry {
    OpType op_type;
    const ConstEvaluator& (*accessor)() noexcept;
};

constexpr EvaluatorEntry kEvaluators[] = {
        {.op_type = OpType::kAdd, .accessor = &detail::GetAddConstEvaluator},
        {.op_type = OpType::kElementwiseMul, .accessor = &detail::GetMulConstEvaluator},
        {.op_type = OpType::kSilu, .accessor = &detail::GetSiluConstEvaluator},
        {.op_type = OpType::kSiluMul, .accessor = &detail::GetSiluMulConstEvaluator},
};

}// namespace

const ConstEvaluator* FindConstEvaluator(OpType op_type) noexcept {
    auto pred = [op_type](const EvaluatorEntry& entry) {
        return entry.op_type == op_type;
    };

    const auto it = std::ranges::find_if(kEvaluators, pred);
    if (it == std::end(kEvaluators)) {
        return nullptr;
    }
    return &it->accessor();
}

}// namespace aethermind
