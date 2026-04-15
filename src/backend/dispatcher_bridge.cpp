//
// Created by richard on 4/15/26.
//

#include "aethermind/backend/dispatcher_bridge.h"

#include <string_view>

namespace aethermind {
namespace {

bool MatchesOperatorName(const OperatorName& op_name,
                         std::string_view canonical_name,
                         std::string_view suffix) noexcept {
    return op_name.name() == canonical_name || op_name.name() == suffix;
}

}// namespace

KernelKey MakeKernelKey(DeviceType device_type, const OperatorName& op_name) {
    return KernelKey{
            .device_type = device_type,
            .op_name = op_name,
    };
}

StatusOr<OpType> ToOpType(const OperatorName& op_name) noexcept {
    if (MatchesOperatorName(op_name, "aethermind::embedding", "embedding")) {
        return OpType::kEmbedding;
    }
    if (MatchesOperatorName(op_name, "aethermind::nn::linear", "linear")) {
        return OpType::kLinear;
    }
    if (MatchesOperatorName(op_name, "aethermind::matmul", "matmul")) {
        return OpType::kMatMul;
    }
    if (MatchesOperatorName(op_name, "aethermind::rms_norm", "rmsnorm")) {
        return OpType::kRMSNorm;
    }
    if (MatchesOperatorName(op_name, "aethermind::rope", "rope")) {
        return OpType::kRoPE;
    }
    if (MatchesOperatorName(op_name, "aethermind::attention_prefill", "attention_prefill")) {
        return OpType::kAttentionPrefill;
    }
    if (MatchesOperatorName(op_name, "aethermind::attention_decode", "attention_decode")) {
        return OpType::kAttentionDecode;
    }
    if (MatchesOperatorName(op_name, "aethermind::silu_mul", "silu_mul")) {
        return OpType::kSiluMul;
    }
    if (MatchesOperatorName(op_name, "aethermind::softmax", "softmax")) {
        return OpType::kSoftmax;
    }
    if (MatchesOperatorName(op_name, "aethermind::argmax", "argmax")) {
        return OpType::kArgMax;
    }

    return Status::NotFound("OperatorName is not mapped to OpType");
}

KernelSelector MakeKernelSelector(DeviceType device_type,
                                  const DataType& activation_dtype,
                                  const DataType& weight_dtype,
                                  WeightFormat weight_format,
                                  IsaLevel isa,
                                  ExecPhase phase) noexcept {
    return KernelSelector{
            .device_type = device_type,
            .activation_dtype = activation_dtype,
            .weight_dtype = weight_dtype,
            .weight_format = weight_format,
            .isa = isa,
            .phase = phase,
    };
}

}// namespace aethermind
