#ifndef AETHERMIND_MODEL_GRAPH_OP_PARAMS_SERDE_H
#define AETHERMIND_MODEL_GRAPH_OP_PARAMS_SERDE_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/op_params.h"

#include <iosfwd>
#include <string_view>
#include <vector>

namespace aethermind {

AM_NODISCARD const char* OpParamsKindName(const OpParams& params) noexcept;

AM_NODISCARD Status SerializeOpParams(const OpParams& params, std::ostream& os);
AM_NODISCARD StatusOr<OpParams> ParseOpParams(std::string_view text);

// Serializes a Reshape target_shape to its canonical textual form, e.g.
// `[@0,@1,32,*]`. Exposed so the canonical Reshape shape spelling is shared
// by SerializeOpParams and DumpOpParams without duplication.
void SerializeReshapeShape(const std::vector<ReshapeDim>& target_shape, std::ostream& os);

// Serializes a Permute permutation to its canonical textual form, e.g.
// `[2,0,1]`. Tokens are unsigned decimals joined with commas and no interior
// whitespace. An empty vector emits `[]` (rank zero). Exposed so serde and
// dump share the canonical spelling without duplication.
void SerializePermutation(const std::vector<uint32_t>& permutation, std::ostream& os);

}// namespace aethermind

#endif
