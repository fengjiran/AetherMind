#ifndef AETHERMIND_GRAPH_GRAPH_OP_BUILDER_H
#define AETHERMIND_GRAPH_GRAPH_OP_BUILDER_H

/// @file graph_op_builder.h
/// @brief High-level helpers for constructing common ModelGraph operator nodes.
///
/// Each `Add*` helper builds a node, registers any required weight or state
/// values, and returns the output value id(s). Builders run operator semantic
/// analysis (shape inference, dtype validation) and emit deferred shape
/// constraints that ModelGraph::ValidateAndTopologicalOrder enforces.
#include "aethermind/base/status.h"
#include "aethermind/graph/graph.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/operators/op_params.h"

#include <optional>
#include <string>

namespace aethermind {

/// @brief Registers an external input tensor and returns its value id.
///
/// @param graph Graph to mutate.
/// @param spec Tensor specification for the input value.
/// @param name Debug name for the new value.
/// @return Value id of the registered input.
AM_NODISCARD GraphValueId AddInput(ModelGraph& graph,
                                   TensorSpec spec,
                                   std::string name = {});

/// @brief Registers a persistent state tensor and returns its value id.
///
/// @param graph Graph to mutate.
/// @param spec Tensor specification for the state tensor.
/// @param binding State binding metadata describing the persistent state kind.
/// @param name Debug name for the new value.
/// @return Value id of the registered state.
AM_NODISCARD GraphValueId AddState(ModelGraph& graph,
                                   TensorSpec spec,
                                   StateBinding binding,
                                   std::string name = {});

/// @brief Builds a Linear (matmul, no bias) node and registers its weight
///        tensor internally.
///
/// The weight is created with shape [out_features, in_features] and dtype
/// `weight_dtype`, bound via `binding`, and named
/// `debug_name + ".weight"`. The node's decoder_layer_index is sourced from
/// `binding.decoder_layer_index`. Input must be ranked with a static, positive
/// last dimension.
///
/// @param graph Graph to mutate.
/// @param input Value id of the input activation.
/// @param out_features Number of output features (weight rows).
/// @param weight_dtype Dtype of the registered weight tensor.
/// @param binding Weight binding metadata (slot, layer index, semantic role).
/// @param name Debug name stem; `.weight` is appended for the weight value.
/// @return Value id of the linear output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddLinear(ModelGraph& graph,
                                 GraphValueId input,
                                 int64_t out_features,
                                 DataType weight_dtype,
                                 WeightBinding binding,
                                 std::string name = {});

/// @brief Builds an RmsNorm node and registers its scale weight tensor
///        internally.
///
/// The weight is created with shape [in_features] and dtype `weight_dtype`,
/// bound via `binding`, and named `name + ".weight"`. The node's
/// decoder_layer_index is sourced from `binding.decoder_layer_index`.
///
/// @param graph Graph to mutate.
/// @param input Value id of the input activation.
/// @param weight_dtype Dtype of the registered scale weight tensor.
/// @param binding Weight binding metadata (slot, layer index, semantic role).
/// @param eps Epsilon added to the variance for numerical stability.
/// @param name Debug name stem; `.weight` is appended for the weight value.
/// @return Value id of the normalized output, or an error status on
///         validation failure.
StatusOr<GraphValueId> AddRmsNorm(ModelGraph& graph,
                                  GraphValueId input,
                                  DataType weight_dtype,
                                  WeightBinding binding,
                                  float eps,
                                  std::string name = {});

/// @brief Builds an Embedding lookup node and registers its embedding table
///        internally.
///
/// The weight is created with shape [vocab_size, embedding_dim] and dtype
/// `weight_dtype`, bound via `binding`, and named `debug_name + ".weight"`.
///
/// @param graph Graph to mutate.
/// @param token_ids Value id of the token indices input.
/// @param vocab_size Vocabulary size (rows of the embedding table).
/// @param embedding_dim Embedding dimension (columns of the table).
/// @param weight_dtype Dtype of the registered embedding table.
/// @param binding Weight binding metadata (slot, layer index, semantic role).
/// @param name Debug name stem; `.weight` is appended for the weight value.
/// @return Value id of the embedding lookup output, or an error status on
///         validation failure.
StatusOr<GraphValueId> AddEmbedding(ModelGraph& graph,
                                    GraphValueId token_ids,
                                    int64_t vocab_size,
                                    int64_t embedding_dim,
                                    DataType weight_dtype,
                                    WeightBinding binding,
                                    std::string name = {});

/// @brief Builds a RoPE node applying rotary position embeddings to Q and K,
///        returning both rotated outputs.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param q Value id of the query tensor.
/// @param k Value id of the key tensor.
/// @param position_ids Value id of the position indices input.
/// @param params RoPE semantic parameters.
/// @param name Debug name for the new node.
/// @return RoPEOutputs holding the rotated q and k value ids, or an error
///         status on validation failure.
StatusOr<RoPEOutputs> AddRoPE(ModelGraph& graph,
                              std::optional<uint32_t> decoder_layer_index,
                              GraphValueId q,
                              GraphValueId k,
                              GraphValueId position_ids,
                              RoPEParams params,
                              std::string name = {});

/// @brief Builds a KVCacheUpdate node appending new K/V tensors to the
///        persistent cache, returning the updated cache state values.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param k_new Value id of the new key tensor to append.
/// @param v_new Value id of the new value tensor to append.
/// @param k_cache Value id of the existing key cache state.
/// @param v_cache Value id of the existing value cache state.
/// @param name Debug name for the new node.
/// @return KVCachePair holding the updated key and value cache value ids, or
///         an error status on validation failure.
StatusOr<KVCachePair> AddKVCacheUpdate(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId k_new,
                                       GraphValueId v_new,
                                       GraphValueId k_cache,
                                       GraphValueId v_cache,
                                       std::string name = {});

/// @brief Builds an Attention node computing scaled dot-product attention
///        over Q, K, V.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param q Value id of the query tensor.
/// @param k Value id of the key tensor.
/// @param v Value id of the value tensor.
/// @param params Attention semantic parameters.
/// @param name Debug name for the new node.
/// @return Value id of the attention output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddAttention(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId q,
                                    GraphValueId k,
                                    GraphValueId v,
                                    AttentionParams params,
                                    std::string name = {});

/// @brief Builds an elementwise Add node with NumPy-style trailing broadcast.
///
/// Both operands must be ranked with matching dtypes. The output shape is
/// inferred via right-aligned broadcast rules; incompatible static dimensions
/// are a fatal check in the builder.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param lhs Value id of the left operand.
/// @param rhs Value id of the right operand.
/// @param name Debug name for the new node.
/// @return Value id of the broadcasted sum output, or an error status on
///         validation failure.
StatusOr<GraphValueId> AddElementwiseAdd(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string name = {});

/// @brief Builds a SiLU-mul node computing silu(gate) * up.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param gate Value id of the gate projection input.
/// @param up Value id of the up projection input.
/// @param name Debug name for the new node.
/// @return Value id of the fused silu-mul output, or an error status on
///         validation failure.
StatusOr<GraphValueId> AddSiluMul(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId gate,
                                  GraphValueId up,
                                  std::string name = {});

/// @brief Builds a SiLU activation node.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param input Value id of the input activation.
/// @param name Debug name for the new node.
/// @return Value id of the activated output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddSilu(ModelGraph& graph,
                               std::optional<uint32_t> decoder_layer_index,
                               GraphValueId input,
                               std::string name = {});

/// @brief Builds an elementwise multiply node.
///
/// lhs and rhs must have matching specs.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param lhs Value id of the left operand.
/// @param rhs Value id of the right operand.
/// @param name Debug name for the new node.
/// @return Value id of the product output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddElementwiseMul(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string name = {});

/// @brief Builds an Argmax node selecting the index of the maximum value
///        along `axis`.
///
/// Output dtype (int64) and reduced shape are derived by operator semantic
/// analysis; the caller must not supply an output spec.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param input Value id of the input tensor.
/// @param axis Axis along which to compute the argmax.
/// @param name Debug name for the new node.
/// @return Value id of the argmax output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddArgmax(ModelGraph& graph,
                                 std::optional<uint32_t> decoder_layer_index,
                                 GraphValueId input,
                                 int64_t axis,
                                 std::string name = {});

/// @brief Builds a semantic Reshape node.
///
/// The output shape is derived entirely by InferReshape from `target_shape`
/// and the input spec; the caller must not supply an output spec. The
/// input's QuantizationSpec is copied to the output value so model-level
/// quantization (e.g. int8 activations) survives reshaping. On failure the
/// graph is left unchanged.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param input Value id of the input tensor.
/// @param target_shape Target shape specification; may include symbolic or
///                     inferred dimensions.
/// @param name Debug name for the new node.
/// @return Value id of the reshaped output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddReshape(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::vector<ReshapeDim> target_shape,
                                  std::string name = {});

/// @brief Builds a semantic Permute node.
///
/// The output shape is derived entirely by InferPermute from `permutation`
/// and the input spec; the caller must not supply an output spec. The
/// input's QuantizationSpec is copied to the output value so model-level
/// quantization (e.g. int8 activations) survives axis permutation. On failure
/// the graph is left unchanged.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param input Value id of the input tensor.
/// @param permutation Axis permutation (e.g. `[2, 0, 1]`).
/// @param name Debug name for the new node.
/// @return Value id of the permuted output, or an error status on validation
///         failure.
StatusOr<GraphValueId> AddPermute(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::vector<uint32_t> permutation,
                                  std::string name = {});

/// @brief Builds a semantic Reorder node.
///
/// The output TensorSpec is an exact copy of the input (dtype, shape, symbol
/// identity); the operator records a fixed physical destination intent —
/// canonical row-major contiguous storage — without exposing a target-format
/// field. The input's QuantizationSpec is copied to the output value. On
/// failure the graph is left unchanged.
///
/// @param graph Graph to mutate.
/// @param decoder_layer_index Decoder layer index assigned to the new node.
/// @param input Value id of the input tensor.
/// @param name Debug name for the new node.
/// @return Value id of the reordered output, or an error status on
///         validation failure.
StatusOr<GraphValueId> AddReorder(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::string name = {});

}// namespace aethermind

#endif
