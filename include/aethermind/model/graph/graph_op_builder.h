#ifndef AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H
#define AETHERMIND_MODEL_GRAPH_GRAPH_OP_BUILDER_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/graph.h"
#include "aethermind/model/graph/graph_types.h"
#include "aethermind/model/graph/op_params.h"

#include <optional>
#include <string>

namespace aethermind {

/// Registers an external input tensor and returns its value id.
AM_NODISCARD GraphValueId AddInput(ModelGraph& graph,
                                   TensorSpec spec,
                                   std::string name = {});

/// Registers a persistent state tensor and returns its value id.
AM_NODISCARD GraphValueId AddState(ModelGraph& graph,
                                   TensorSpec spec,
                                   StateBinding binding,
                                   std::string name = {});

/// Builds a Linear (matmul, no bias) node and registers its weight tensor
/// internally. The weight is created with shape [out_features, in_features]
/// and dtype `weight_dtype`, bound via `binding`, and named
/// `debug_name + ".weight"`. The node's decoder_layer_index is sourced from
/// `binding.decoder_layer_index`. Input must be ranked with a static, positive
/// last dimension.
StatusOr<GraphValueId> AddLinear(ModelGraph& graph,
                                 GraphValueId input,
                                 int64_t out_features,
                                 DataType weight_dtype,
                                 WeightBinding binding,
                                 std::string name = {});

/// Builds an RmsNorm node and registers its scale weight tensor internally.
/// The weight is created with shape [in_features] and dtype `weight_dtype`,
/// bound via `binding`, and named `name + ".weight"`. The node's
/// decoder_layer_index is sourced from `binding.decoder_layer_index`.
StatusOr<GraphValueId> AddRmsNorm(ModelGraph& graph,
                                  GraphValueId input,
                                  DataType weight_dtype,
                                  WeightBinding binding,
                                  float eps,
                                  std::string name = {});

/// Builds an Embedding lookup node and registers its embedding table internally.
/// The weight is created with shape [vocab_size, embedding_dim] and dtype
/// `weight_dtype`, bound via `binding`, and named `debug_name + ".weight"`.
StatusOr<GraphValueId> AddEmbedding(ModelGraph& graph,
                                    GraphValueId token_ids,
                                    int64_t vocab_size,
                                    int64_t embedding_dim,
                                    DataType weight_dtype,
                                    WeightBinding binding,
                                    std::string name = {});

/// Builds a RoPE node applying rotary position embeddings to Q and K,
/// returning both rotated outputs.
StatusOr<RoPEOutputs> AddRoPE(ModelGraph& graph,
                              std::optional<uint32_t> decoder_layer_index,
                              GraphValueId q,
                              GraphValueId k,
                              GraphValueId position_ids,
                              RoPEParams params,
                              std::string name = {});

/// Builds a KVCacheUpdate node appending new K/V tensors to the persistent
/// cache, returning the updated cache state values.
StatusOr<KVCachePair> AddKVCacheUpdate(ModelGraph& graph,
                                       std::optional<uint32_t> decoder_layer_index,
                                       GraphValueId k_new,
                                       GraphValueId v_new,
                                       GraphValueId k_cache,
                                       GraphValueId v_cache,
                                       std::string name = {});

/// Builds an Attention node computing scaled dot-product attention over Q, K, V.
StatusOr<GraphValueId> AddAttention(ModelGraph& graph,
                                    std::optional<uint32_t> decoder_layer_index,
                                    GraphValueId q,
                                    GraphValueId k,
                                    GraphValueId v,
                                    AttentionParams params,
                                    std::string name = {});

/// Builds an elementwise Add node with NumPy-style trailing broadcast.
///
/// Both operands must be ranked with matching dtypes. The output shape is
/// inferred via right-aligned broadcast rules; incompatible static dimensions
/// are a fatal check in the builder.
StatusOr<GraphValueId> AddElementwiseAdd(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string name = {});

/// Builds a SiLU-mul node computing silu(gate) * up.
StatusOr<GraphValueId> AddSiluMul(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId gate,
                                  GraphValueId up,
                                  std::string name = {});

/// Builds a SiLU activation node.
StatusOr<GraphValueId> AddSilu(ModelGraph& graph,
                               std::optional<uint32_t> decoder_layer_index,
                               GraphValueId input,
                               std::string name = {});

/// Builds an elementwise multiply node. lhs and rhs must have matching specs.
StatusOr<GraphValueId> AddElementwiseMul(ModelGraph& graph,
                                         std::optional<uint32_t> decoder_layer_index,
                                         GraphValueId lhs,
                                         GraphValueId rhs,
                                         std::string name = {});

/// Builds an Argmax node selecting the index of the maximum value along `axis`.
/// Output dtype (int64) and reduced shape are derived by operator semantic
/// analysis; the caller must not supply an output spec.
StatusOr<GraphValueId> AddArgmax(ModelGraph& graph,
                                 std::optional<uint32_t> decoder_layer_index,
                                 GraphValueId input,
                                 int64_t axis,
                                 std::string name = {});

/// Builds a semantic Reshape node. The output shape is derived entirely by
/// InferReshape from `target_shape` and the input spec; the caller must not
/// supply an output spec. The input's QuantizationSpec is copied to the
/// output value so model-level quantization (e.g. int8 activations) survives
/// reshaping. On failure the graph is left unchanged.
StatusOr<GraphValueId> AddReshape(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::vector<ReshapeDim> target_shape,
                                  std::string name = {});

/// Builds a semantic Permute node. The output shape is derived entirely by
/// InferPermute from `permutation` and the input spec; the caller must not
/// supply an output spec. The input's QuantizationSpec is copied to the
/// output value so model-level quantization (e.g. int8 activations) survives
/// axis permutation. On failure the graph is left unchanged.
StatusOr<GraphValueId> AddPermute(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::vector<uint32_t> permutation,
                                  std::string name = {});

/// Builds a semantic Reorder node. The output TensorSpec is an exact copy of
/// the input (dtype, shape, symbol identity); the operator records a fixed
/// physical destination intent — canonical row-major contiguous storage —
/// without exposing a target-format field. The input's QuantizationSpec is
/// copied to the output value. On failure the graph is left unchanged.
StatusOr<GraphValueId> AddReorder(ModelGraph& graph,
                                  std::optional<uint32_t> decoder_layer_index,
                                  GraphValueId input,
                                  std::string name = {});

}// namespace aethermind

#endif
