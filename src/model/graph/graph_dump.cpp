#include "aethermind/model/graph/graph_dump.h"
#include "aethermind/dtypes/data_type.h"
#include "utils/variant_utils.h"

#include <ostream>
#include <string_view>
#include <variant>

namespace aethermind {
namespace {

const char* QuantizationKindName(QuantizationKind kind) noexcept {
    switch (kind) {
        case QuantizationKind::kNone:
            return "none";
        case QuantizationKind::kInt8:
            return "int8";
        case QuantizationKind::kInt4:
            return "int4";
    }
    return "unknown";
}

void DumpGraphValueId(GraphValueId id, std::ostream& os) {
    os << 'v' << id.index;
}

void DumpGraphNodeId(GraphNodeId id, std::ostream& os) {
    os << 'n' << id.index;
}

void DumpTensorSpec(const TensorSpec& spec, std::ostream& os) {
    os << ToString(spec.dtype) << spec.shape;
}

void DumpWeightBinding(const WeightBinding& binding, std::ostream& os) {
    os << "slot=" << ToString(binding.slot)
       << ", semantic=" << ToString(binding.semantic_role);
    if (binding.decoder_layer_index.has_value()) {
        os << ", layer=" << *binding.decoder_layer_index;
    } else {
        os << ", layer=<none>";
    }
}

void DumpStateBinding(const StateBinding& binding, std::ostream& os) {
    auto visitor = overloaded{
            [&](const KVCacheStateBinding& kv) {
                os << "kv_cache(layer=" << kv.decoder_layer_index
                   << ", slot=" << ToString(kv.slot) << ')';
            },
            [&](const DecodeStateBinding&) {
                os << "decode_state";
            },
            [&](const StreamingStateBinding&) {
                os << "streaming_state";
            },
    };
    std::visit(visitor, binding);
}

void DumpConstantBinding(const ConstantBinding& binding, std::ostream& os) {
    os << "name=" << binding.name;
    const std::size_t bytes = binding.inline_data ? binding.inline_data->size() : 0U;
    os << ", inline_data=" << bytes << "B";
}

void DumpPayload(const GraphValuePayload& payload, std::ostream& os) {
    auto visitor = overloaded{
            [&](const std::monostate&) {
                os << "monostate";
            },
            [&](const ModelInputValue&) {
                os << "model_input";
            },
            [&](const ActivationValue&) {
                os << "activation";
            },
            [&](const WeightValue& weight) {
                os << "weight(";
                DumpWeightBinding(weight.binding, os);
                os << ')';
            },
            [&](const ConstantValue& constant) {
                os << "constant(";
                DumpConstantBinding(constant.binding, os);
                os << ')';
            },
            [&](const StateValue& state) {
                os << "state(";
                DumpStateBinding(state.binding, os);
                os << ')';
            },
    };
    std::visit(visitor, payload);
}

void DumpValueIdList(const std::vector<GraphValueId>& values, std::ostream& os) {
    os << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        DumpGraphValueId(values[i], os);
    }
    os << ']';
}

void DumpEmptyParams(std::string_view name, std::ostream& os) {
    os << name << "{}";
}

}// namespace

const char* ToString(ParameterSlot slot) noexcept {
    switch (slot) {
        case ParameterSlot::kKernel:
            return "Kernel";
        case ParameterSlot::kBias:
            return "Bias";
        case ParameterSlot::kScale:
            return "Scale";
        case ParameterSlot::kShift:
            return "Shift";
        case ParameterSlot::kEmbeddingTable:
            return "EmbeddingTable";
    }
    return "UnknownParameterSlot";
}

const char* ToString(TransformerWeightRole role) noexcept {
    switch (role) {
        case TransformerWeightRole::kTokenEmbedding:
            return "TokenEmbedding";
        case TransformerWeightRole::kInputNorm:
            return "InputNorm";
        case TransformerWeightRole::kAttentionQ:
            return "AttentionQ";
        case TransformerWeightRole::kAttentionK:
            return "AttentionK";
        case TransformerWeightRole::kAttentionV:
            return "AttentionV";
        case TransformerWeightRole::kAttentionO:
            return "AttentionO";
        case TransformerWeightRole::kMlpGate:
            return "MlpGate";
        case TransformerWeightRole::kMlpUp:
            return "MlpUp";
        case TransformerWeightRole::kMlpDown:
            return "MlpDown";
        case TransformerWeightRole::kPostAttentionNorm:
            return "PostAttentionNorm";
        case TransformerWeightRole::kFinalNorm:
            return "FinalNorm";
        case TransformerWeightRole::kLmHead:
            return "LmHead";
        case TransformerWeightRole::kMoERouter:
            return "MoERouter";
    }
    return "UnknownTransformerWeightRole";
}

const char* ToString(const ModelSemanticRole& role) noexcept {
    auto visitor = overloaded{
            [](const std::monostate&) noexcept {
                return "<none>";
            },
            [](TransformerWeightRole transformer_role) noexcept {
                return ToString(transformer_role);
            },
    };
    return std::visit(visitor, role);
}

const char* ToString(KVCacheSlot slot) noexcept {
    switch (slot) {
        case KVCacheSlot::kKey:
            return "Key";
        case KVCacheSlot::kValue:
            return "Value";
    }
    return "UnknownKVCacheSlot";
}

const char* GraphValuePayloadKindName(const GraphValuePayload& payload) noexcept {
    auto visitor = overloaded{
            [](const std::monostate&) noexcept {
                return "monostate";
            },
            [](const ModelInputValue&) noexcept {
                return "model_input";
            },
            [](const ActivationValue&) noexcept {
                return "activation";
            },
            [](const WeightValue&) noexcept {
                return "weight";
            },
            [](const ConstantValue&) noexcept {
                return "constant";
            },
            [](const StateValue&) noexcept {
                return "state";
            },
    };
    return std::visit(visitor, payload);
}

void DumpOpParams(const OpParams& params, std::ostream& os) {
    auto visitor = overloaded{
            [&](const std::monostate&) {
                os << "monostate{}";
            },
            [&](const EmbeddingParams&) {
                DumpEmptyParams("EmbeddingParams", os);
            },
            [&](const RmsNormParams& p) {
                os << "RmsNormParams{eps=" << p.eps << '}';
            },
            [&](const LinearParams&) {
                DumpEmptyParams("LinearParams", os);
            },
            [&](const RoPEParams& p) {
                os << "RoPEParams{head_dim=" << p.head_dim
                   << ", num_attention_heads=" << p.num_attention_heads
                   << ", num_key_value_heads=" << p.num_key_value_heads
                   << ", max_position_embeddings=" << p.max_position_embeddings
                   << ", theta=" << p.theta
                   << ", scaling_factor=";
                if (p.scaling_factor.has_value()) {
                    os << *p.scaling_factor;
                } else {
                    os << "<none>";
                }
                os << ", scaling_type=" << ToString(p.scaling_type) << '}';
            },
            [&](const MatMulParams& p) {
                os << "MatMulParams{transpose_rhs=" << (p.transpose_rhs ? "true" : "false") << '}';
            },
            [&](const SoftmaxParams& p) {
                os << "SoftmaxParams{axis=" << p.axis << '}';
            },
            [&](const AddParams&) {
                DumpEmptyParams("AddParams", os);
            },
            [&](const SiluParams&) {
                DumpEmptyParams("SiluParams", os);
            },
            [&](const SiluMulParams&) {
                DumpEmptyParams("SiluMulParams", os);
            },
            [&](const ElementwiseMulParams&) {
                DumpEmptyParams("ElementwiseMulParams", os);
            },
            [&](const KVCacheUpdateParams&) {
                DumpEmptyParams("KVCacheUpdateParams", os);
            },
            [&](const AttentionParams& p) {
                os << "AttentionParams{num_attention_heads=" << p.num_attention_heads
                   << ", num_key_value_heads=" << p.num_key_value_heads
                   << ", head_dim=" << p.head_dim << '}';
            },
            [&](const ArgmaxParams& p) {
                os << "ArgmaxParams{axis=" << p.axis << '}';
            },
    };
    std::visit(visitor, params);
}

void DumpGraph(const ModelGraph& graph, std::ostream& os) {
    os << "ModelGraph\n";

    os << "inputs:\n";
    for (const auto& input: graph.GetInputs()) {
        os << "  ";
        DumpGraphValueId(input.value, os);
        os << ", name=" << graph.GetValue(input.value).name << '\n';
    }

    os << "outputs:\n";
    for (const auto& output: graph.GetOutputs()) {
        os << "  ";
        DumpGraphValueId(output.value, os);
        os << ", name=" << graph.GetValue(output.value).name << '\n';
    }

    os << "values:\n";
    const std::span<const GraphValue> values = graph.GetValues();
    for (size_t i = 0; i < values.size(); ++i) {
        const GraphValue& value = values[i];
        os << "  v" << i << ", kind=" << GraphValuePayloadKindName(value.payload) << ", spec=";
        DumpTensorSpec(value.spec, os);
        os << ", payload=";
        DumpPayload(value.payload, os);
        os << ", producer=";
        if (value.producer.has_value()) {
            DumpGraphNodeId(*value.producer, os);
        } else {
            os << "<none>";
        }
        if (!value.name.empty()) {
            os << ", debug_name=" << value.name;
        }
        if (value.quantization.kind != QuantizationKind::kNone) {
            os << ", quant=" << QuantizationKindName(value.quantization.kind)
               << ", group_size=" << value.quantization.group_size;
        }
        os << '\n';
    }

    os << "nodes:\n";
    const std::span<const GraphNode> nodes = graph.GetNodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode& node = nodes[i];
        os << "  n" << i << ", op=" << ToString(node.op_type);
        if (node.decoder_layer_index.has_value()) {
            os << ", layer=" << *node.decoder_layer_index;
        }
        os << ", inputs=";
        DumpValueIdList(node.inputs, os);
        os << ", outputs=";
        DumpValueIdList(node.outputs, os);
        os << ", params=";
        DumpOpParams(node.op_params, os);
        if (!node.attrs.bytes.empty()) {
            os << ", attrs=" << node.attrs.bytes.size() << "B";
        }
        if (!node.name.empty()) {
            os << ", debug_name=" << node.name;
        }
        os << '\n';
    }
}

}// namespace aethermind
