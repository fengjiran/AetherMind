#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H

#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/graph_types.h"

#include <optional>
#include <string>

namespace aethermind {

/// Registers an external input tensor and returns its value id.
AM_NODISCARD GraphValueId AddInput(ModelGraph& graph,
                                   TensorSpec spec,
                                   std::string debug_name = {});

/// Registers a persistent state tensor and returns its value id.
AM_NODISCARD GraphValueId AddState(ModelGraph& graph,
                                   TensorSpec spec,
                                   StateBinding binding,
                                   std::string debug_name = {});

/// Builds a Linear (matmul, no bias) node and registers its weight tensor
/// internally. The weight is created with shape [out_features, in_features]
/// and dtype `weight_dtype`, bound via `binding`, and named
/// `debug_name + ".weight"`. The node's decoder_layer_index is sourced from
/// `binding.decoder_layer_index`. Input must be ranked with a static, positive
/// last dimension.
AM_NODISCARD GraphValueId AddLinear(ModelGraph& graph,
                                    GraphValueId input,
                                    int64_t out_features,
                                    DataType weight_dtype,
                                    WeightBinding binding,
                                    std::string debug_name = {});

/// Builds an RmsNorm node and registers its scale weight tensor internally.
/// The weight is created with shape [in_features] and dtype `weight_dtype`,
/// bound via `binding`, and named `debug_name + ".weight"`. The node's
/// decoder_layer_index is sourced from `binding.decoder_layer_index`.
AM_NODISCARD GraphValueId AddRmsNorm(ModelGraph& graph,
                                     GraphValueId input,
                                     DataType weight_dtype,
                                     WeightBinding binding,
                                     float eps,
                                     std::string debug_name = {});

/// Builds an Embedding lookup node and registers its embedding table internally.
/// The weight is created with shape [vocab_size, embedding_dim] and dtype
/// `weight_dtype`, bound via `binding`, and named `debug_name + ".weight"`.
AM_NODISCARD GraphValueId AddEmbedding(ModelGraph& graph,
                                       GraphValueId token_ids,
                                       int64_t vocab_size,
                                       int64_t embedding_dim,
                                       DataType weight_dtype,
                                       WeightBinding binding,
                                       std::string debug_name = {});

/// Builds a RoPE node applying rotary position embeddings to Q and K,
/// returning both rotated outputs.
AM_NODISCARD RoPEOutputs AddRoPE(ModelGraph& graph,
                                 std::optional<uint32_t> decoder_layer_index,
                                 GraphValueId q,
                                 GraphValueId k,
                                 GraphValueId position_ids,
                                 RoPEParams params,
                                 std::string debug_name = {});

/// Builds a KVCacheUpdate node appending new K/V tensors to the persistent
/// cache, returning the updated cache state values.
AM_NODISCARD KVCachePair AddKVCacheUpdate(ModelGraph& graph,
                                          std::optional<uint32_t> decoder_layer_index,
                                          GraphValueId k_new,
                                          GraphValueId v_new,
                                          GraphValueId k_cache,
                                          GraphValueId v_cache,
                                          std::string debug_name = {});

/// Builds an Attention node computing scaled dot-product attention over Q, K, V.
AM_NODISCARD GraphValueId AddAttention(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId q,
                                       GraphValueId k,
                                       GraphValueId v,
                                       AttentionParams params,
                                       std::string debug_name = {});

/// Builds an elementwise Add node, typically used for residual connections.
AM_NODISCARD GraphValueId AddElementwiseAdd(ModelGraph& graph,
                                            std::optional<uint32_t> decoder_layer_index,
                                            GraphValueId lhs,
                                            GraphValueId rhs,
                                            std::string debug_name = {});

/// Builds a SiLU-mul node computing silu(gate) * up.
AM_NODISCARD GraphValueId AddSiluMul(ModelGraph& graph,
                                     std::optional<uint32_t> decoder_layer_index,
                                     GraphValueId gate,
                                     GraphValueId up,
                                     std::string debug_name = {});

/// Builds an Argmax node selecting the index of the maximum value along `axis`.
AM_NODISCARD GraphValueId AddArgmax(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId input,
                                    TensorSpec output_spec,
                                    int64_t axis,
                                    std::string debug_name = {});

}// namespace aethermind

#endif
