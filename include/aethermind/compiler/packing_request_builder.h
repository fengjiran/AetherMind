#ifndef AETHERMIND_COMPILER_PACKING_REQUEST_BUILDER_H
#define AETHERMIND_COMPILER_PACKING_REQUEST_BUILDER_H

/// @file packing_request_builder.h
/// @brief Derives weight-packing requests from a compiler artifact.

#include "aethermind/base/status.h"
#include "aethermind/compiler/lowered_graph.h"
#include "aethermind/model/resolved_model_weights.h"
#include "aethermind/model/weight_prepack_planner.h"

#include <vector>

namespace aethermind {

/// @brief Maps a finalized artifact's kWeight values to packing requests.
///
/// Pure data mapping: reads only the lowered graph and resolved weights, and
/// produces WeightPrepackPlanner::Request entries carrying the artifact
/// identity (source_id + value_index), the logical binding, its raw weight,
/// and the step selector. It never touches a backend; packing is executed by
/// WeightPrepackPlanner::PrepackAndStore.
///
/// @param lowered Finalized compiler artifact.
/// @param resolved Resolved raw weights backing the artifact.
/// @return One request per kWeight input value, or an error when a weight
///         value lacks a resolvable raw weight or binding role.
AM_NODISCARD StatusOr<std::vector<WeightPrepackPlanner::Request>>
BuildWeightPackingRequests(const LoweredGraph& lowered,
                           const ResolvedModelWeights& resolved);

} // namespace aethermind

#endif
