#include "aethermind/model/graph/graph_dump.h"
#include "aethermind/dtypes/data_type.h"
#include "utils/variant_utils.h"

#include <ostream>
#include <string_view>
#include <variant>

namespace aethermind {
namespace {

void DumpGraphValueId(GraphValueId id, std::ostream& os) {
    os << 'v' << id.index;
}

void DumpGraphNodeId(GraphNodeId id, std::ostream& os) {
    os << 'n' << id.index;
}

void DumpShape(const TensorSpec& spec, std::ostream& os) {
    os << DataTypeToString(spec.dtype) << spec.shape;
}

void DumpWeightBinding(const WeightBinding& binding, std::ostream& os) {
    os << "role=" << ToString(binding.role);
    if (binding.decoder_layer_index.has_value()) {
        os << " layer=" << *binding.decoder_layer_index;
    } else {
        os << " layer=<none>";
    }
}

void DumpStateBinding(const StateBinding& binding, std::ostream& os) {
    std::visit(overloaded{
                       [&](const KVCacheStateBinding& kv) {
                           os << "kv_cache(layer=" << kv.decoder_layer_index
                              << " slot=" << ToString(kv.slot) << ')';
                       },
                       [&](const DecodeStateBinding&) {
                           os << "decode_state";
                       },
                       [&](const StreamingStateBinding&) {
                           os << "streaming_state";
                       },
               },
               binding);
}

void DumpPayload(const GraphValuePayload& payload, std::ostream& os) {
    std::visit(overloaded{
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
                       [&](const StateValue& state) {
                           os << "state(";
                           DumpStateBinding(state.binding, os);
                           os << ')';
                       },
               },
               payload);
}

void DumpInputList(const std::vector<GraphValueId>& values, std::ostream& os) {
    os << '[';
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            os << ", ";
        }
        DumpGraphValueId(values[i], os);
    }
    os << ']';
}

void DumpCommonEmptyParams(std::string_view name, std::ostream& os) {
    os << name << "{}";
}

}// namespace

const char* ToString(WeightRole role) noexcept {
    switch (role) {
        case WeightRole::kTokenEmbedding:
            return "TokenEmbedding";
        case WeightRole::kAttentionQ:
            return "AttentionQ";
        case WeightRole::kAttentionK:
            return "AttentionK";
        case WeightRole::kAttentionV:
            return "AttentionV";
        case WeightRole::kAttentionO:
            return "AttentionO";
        case WeightRole::kMlpGate:
            return "MlpGate";
        case WeightRole::kMlpUp:
            return "MlpUp";
        case WeightRole::kMlpDown:
            return "MlpDown";
        case WeightRole::kInputNorm:
            return "InputNorm";
        case WeightRole::kPostAttentionNorm:
            return "PostAttentionNorm";
        case WeightRole::kFinalNorm:
            return "FinalNorm";
        case WeightRole::kLmHead:
            return "LmHead";
    }
    return "UnknownWeightRole";
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
            [](const std::monostate&) noexcept { return "monostate"; },
            [](const ModelInputValue&) noexcept { return "model_input"; },
            [](const ActivationValue&) noexcept { return "activation"; },
            [](const WeightValue&) noexcept { return "weight"; },
            [](const StateValue&) noexcept { return "state"; },
    };
    return std::visit(visitor, payload);
}

Status DumpOpParams(const OpParams& params, std::ostream& os) {
    auto visitor = overloaded{
            [&](const std::monostate&) { os << "monostate{}"; },
            [&](const EmbeddingParams&) { DumpCommonEmptyParams("EmbeddingParams", os); },
            [&](const RmsNormParams& p) { os << "RmsNormParams{eps=" << p.eps << '}'; },
            [&](const LinearParams&) { DumpCommonEmptyParams("LinearParams", os); },
            [&](const RoPEParams& p) {
                os << "RoPEParams{head_dim=" << p.head_dim
                   << " num_attention_heads=" << p.num_attention_heads
                   << " num_key_value_heads=" << p.num_key_value_heads
                   << " max_position_embeddings=" << p.max_position_embeddings
                   << " theta=" << p.theta
                   << " scaling_factor=";
                if (p.scaling_factor.has_value()) {
                    os << *p.scaling_factor;
                } else {
                    os << "<none>";
                }
                os << " scaling_type=" << ToString(p.scaling_type) << '}';
            },
            [&](const MatMulParams& p) {
                os << "MatMulParams{transpose_rhs=" << (p.transpose_rhs ? "true" : "false") << '}';
            },
            [&](const SoftmaxParams& p) { os << "SoftmaxParams{axis=" << p.axis << '}'; },
            [&](const AddParams&) { DumpCommonEmptyParams("AddParams", os); },
            [&](const SiluMulParams&) { DumpCommonEmptyParams("SiluMulParams", os); },
            [&](const KVCacheUpdateParams&) { DumpCommonEmptyParams("KVCacheUpdateParams", os); },
            [&](const AttentionParams& p) {
                os << "AttentionParams{num_attention_heads=" << p.num_attention_heads
                   << " num_key_value_heads=" << p.num_key_value_heads
                   << " head_dim=" << p.head_dim << '}';
            },
            [&](const ArgmaxParams& p) { os << "ArgmaxParams{axis=" << p.axis << '}'; },
    };
    std::visit(visitor, params);
    return Status::Ok();
}

Status DumpGraph(const ModelGraph& graph, std::ostream& os) {
    os << "ModelGraph\n";

    os << "inputs:\n";
    for (const ModelGraph::Input& input: graph.GetInputs()) {
        os << "  ";
        DumpGraphValueId(input.value, os);
        os << " name=" << input.name << '\n';
    }

    os << "outputs:\n";
    for (const ModelGraph::Output& output: graph.GetOutputs()) {
        os << "  ";
        DumpGraphValueId(output.value, os);
        os << " name=" << output.name << '\n';
    }

    os << "values:\n";
    const std::span<const GraphValue> values = graph.GetValues();
    for (size_t i = 0; i < values.size(); ++i) {
        const GraphValue& value = values[i];
        os << "  v" << i << " kind=" << GraphValuePayloadKindName(value.payload) << " spec=";
        DumpShape(value.spec, os);
        os << " payload=";
        DumpPayload(value.payload, os);
        os << " producer=";
        if (value.producer.has_value()) {
            DumpGraphNodeId(*value.producer, os);
        } else {
            os << "<none>";
        }
        if (!value.debug_name.empty()) {
            os << " debug_name=" << value.debug_name;
        }
        os << '\n';
    }

    os << "nodes:\n";
    const std::span<const GraphNode> nodes = graph.GetNodes();
    for (size_t i = 0; i < nodes.size(); ++i) {
        const GraphNode& node = nodes[i];
        os << "  n" << i << " op=" << ToString(node.op_type);
        if (node.decoder_layer_index.has_value()) {
            os << " layer=" << *node.decoder_layer_index;
        }
        os << " inputs=";
        DumpInputList(node.inputs, os);
        os << " outputs=";
        DumpInputList(node.outputs, os);
        os << " params=";
        AM_RETURN_IF_ERROR(DumpOpParams(node.op_params, os));
        if (!node.debug_name.empty()) {
            os << " debug_name=" << node.debug_name;
        }
        os << '\n';
    }

    return Status::Ok();
}

}// namespace aethermind
