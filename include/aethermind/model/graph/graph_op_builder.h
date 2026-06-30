#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H

#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/graph_types.h"

#include <optional>
#include <string>

namespace aethermind {

struct RoPEOutputs {
    GraphValueId q{};
    GraphValueId k{};
};

struct KVCacheUpdateOutputs {
    GraphValueId k{};
    GraphValueId v{};
};

/// Registers an external input tensor and returns its value id.
AM_NODISCARD GraphValueId AddInput(ModelGraph& graph,
                                   TensorSpec spec,
                                   std::string debug_name = {});

/// Registers a persistent state tensor and returns its value id.
AM_NODISCARD GraphValueId AddState(ModelGraph& graph,
                                   TensorSpec spec,
                                   StateBinding binding,
                                   std::string debug_name = {});

AM_NODISCARD GraphValueId AddLinear(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId input,
                                    GraphValueId weight,
                                    TensorSpec output_spec,
                                    std::string debug_name = {});

AM_NODISCARD GraphValueId AddRmsNorm(ModelGraph& graph,
                                     std::optional<uint32_t> decoder_layer_index,
                                     GraphValueId input,
                                     GraphValueId weight,
                                     TensorSpec output_spec,
                                     float eps,
                                     std::string debug_name = {});

AM_NODISCARD GraphValueId AddEmbedding(ModelGraph& graph,
                                       GraphValueId token_ids,
                                       GraphValueId weight,
                                       TensorSpec output_spec,
                                       std::string debug_name = {});

AM_NODISCARD RoPEOutputs AddRoPE(ModelGraph& graph,
                                 std::optional<uint32_t> decoder_layer_index,
                                 GraphValueId q,
                                 GraphValueId k,
                                 GraphValueId position_ids,
                                 TensorSpec q_output_spec,
                                 TensorSpec k_output_spec,
                                 RoPEParams params,
                                 std::string debug_name = {});

AM_NODISCARD KVCacheUpdateOutputs AddKVCacheUpdate(ModelGraph& graph,
                                                   std::optional<uint32_t> decoder_layer_index,
                                                   GraphValueId k_new,
                                                   GraphValueId v_new,
                                                   GraphValueId k_cache,
                                                   GraphValueId v_cache,
                                                   TensorSpec k_output_spec,
                                                   TensorSpec v_output_spec,
                                                   StateBinding k_binding,
                                                   StateBinding v_binding,
                                                   std::string debug_name = {});

AM_NODISCARD GraphValueId AddAttention(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId q,
                                       GraphValueId k,
                                       GraphValueId v,
                                       TensorSpec output_spec,
                                       AttentionParams params,
                                       std::string debug_name = {});

AM_NODISCARD GraphValueId AddElementwiseAdd(ModelGraph& graph,
                                            std::optional<uint32_t> decoder_layer_index,
                                            GraphValueId lhs,
                                            GraphValueId rhs,
                                            TensorSpec output_spec,
                                            std::string debug_name = {});

AM_NODISCARD GraphValueId AddSiluMul(ModelGraph& graph,
                                     std::optional<uint32_t> decoder_layer_index,
                                     GraphValueId gate,
                                     GraphValueId up,
                                     TensorSpec output_spec,
                                     std::string debug_name = {});

AM_NODISCARD GraphValueId AddArgmax(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId input,
                                    TensorSpec output_spec,
                                    int64_t axis,
                                    std::string debug_name = {});

}// namespace aethermind

#endif
