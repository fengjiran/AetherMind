#ifndef AETHERMIND_MODEL_TRANSFORMER_GRAPH_COMMON_INTERNAL_H
#define AETHERMIND_MODEL_TRANSFORMER_GRAPH_COMMON_INTERNAL_H

/// @file transformer_graph_common.h
/// @brief Family-independent assembly helpers shared by per-family graph
/// builders (debug names, KV-cache spec, layer prefixes). Internal to the
/// model module.

#include "aethermind/dtypes/data_type.h"
#include "aethermind/graph/graph_types.h"
#include "aethermind/shape_inference/tensor_spec.h"

#include <cstdint>
#include <optional>
#include <string>

namespace aethermind::detail {

/// @brief Prefix for per-layer debug names, e.g. "layers.3.".
std::string LayerPrefix(uint32_t layer);

/// @brief Canonical checkpoint name for a transformer weight role.
///
/// Single authority for role → debug-name mapping consumed by every
/// per-family builder; keeps graph dumps readable regardless of family.
/// Layer-scoped roles require `layer`, global roles require nullopt.
std::string WeightDebugName(TransformerWeightRole role,
                            std::optional<uint32_t> layer);

/// @brief KV-cache tensor spec shared by attention-heavy transformer families.
TensorSpec KVCacheTensorSpec(DataType dtype, int64_t num_kv_heads,
                             ShapeSymbol cache_len, int64_t head_dim);

} // namespace aethermind::detail

#endif // AETHERMIND_MODEL_TRANSFORMER_GRAPH_COMMON_INTERNAL_H