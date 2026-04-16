//
// Created by Richard on 4/16/26.
//

#include "aethermind/operators/op_type.h"

namespace aethermind {

const char* ToString(OpType op_type) noexcept {
    switch (op_type) {
        case OpType::kUnknown:
            return "Unknown";
        case OpType::kEmbedding:
            return "Embedding";
        case OpType::kLinear:
            return "Linear";
        case OpType::kMatMul:
            return "MatMul";
        case OpType::kRMSNorm:
            return "RMSNorm";
        case OpType::kRoPE:
            return "RoPE";
        case OpType::kAttentionPrefill:
            return "AttentionPrefill";
        case OpType::kAttentionDecode:
            return "AttentionDecode";
        case OpType::kSiluMul:
            return "SiluMul";
        case OpType::kSoftmax:
            return "Softmax";
        case OpType::kArgMax:
            return "ArgMax";
        default:
            return "Unknown";
    }
}

}// namespace aethermind