#ifndef AETHERMIND_MODEL_GRAPH_OP_PARAMS_SERDE_H
#define AETHERMIND_MODEL_GRAPH_OP_PARAMS_SERDE_H

#include "aethermind/base/status.h"
#include "aethermind/model/graph/op_params.h"

#include <iosfwd>
#include <string_view>

namespace aethermind {

AM_NODISCARD const char* OpParamsKindName(const OpParams& params) noexcept;

AM_NODISCARD Status SerializeOpParams(const OpParams& params, std::ostream& os);
AM_NODISCARD StatusOr<OpParams> ParseOpParams(std::string_view text);

}// namespace aethermind

#endif
