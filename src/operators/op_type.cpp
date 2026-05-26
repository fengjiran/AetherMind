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
        case OpType::kRmsNorm:
            return "RmsNorm";
        case OpType::kLinear:
            return "Linear";
        case OpType::kMatMul:
            return "MatMul";
        case OpType::kRoPE:
            return "RoPE";
        case OpType::kAttention:
            return "Attention";
        case OpType::kSilu:
            return "Silu";
        case OpType::kSiluMul:
            return "SiluMul";
        case OpType::kElementwiseMul:
            return "ElementwiseMul";
        case OpType::kAdd:
            return "Add";
        case OpType::kSoftmax:
            return "Softmax";
        case OpType::kArgmax:
            return "Argmax";
        default:
            return "Unknown";
    }
}

}// namespace aethermind
