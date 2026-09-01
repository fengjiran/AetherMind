#ifndef AETHERMIND_OPERATORS_OP_PARAMS_SERDE_H
#define AETHERMIND_OPERATORS_OP_PARAMS_SERDE_H

/// @file op_params_serde.h
/// @brief Canonical serialization and parsing of OpParams variants.
///
/// Provides a stable textual form for OpParams used by both serialization
/// (SerializeOpParams) and diagnostics (DumpOpParams). Shared canonical
/// spellings for Reshape shapes and Permute permutations are exposed so the
/// two paths never diverge.
#include "aethermind/base/status.h"
#include "aethermind/operators/op_params.h"

#include <iosfwd>
#include <string_view>
#include <vector>

namespace aethermind {

/// @brief Returns the canonical kind name of the active OpParams alternative.
///
/// @param params Operator parameters to inspect.
/// @return Stable kind name (e.g. "Embedding", "RmsNorm", "Linear").
AM_NODISCARD const char* OpParamsKindName(const OpParams& params) noexcept;

/// @brief Serializes an OpParams variant to `os` in canonical textual form.
///
/// @param params Operator parameters to serialize.
/// @param os Output stream to write to.
/// @return Status::Ok on success, or an error status if writing fails.
Status SerializeOpParams(const OpParams& params, std::ostream& os);

/// @brief Parses an OpParams variant from its canonical textual form.
///
/// @param text Canonical textual form as produced by SerializeOpParams.
/// @return Parsed OpParams, or an error status on malformed input.
StatusOr<OpParams> ParseOpParams(std::string_view text);

/// @brief Serializes a Reshape target_shape to its canonical textual form.
///
/// Example: `[@0,@1,32,*]`. Exposed so the canonical Reshape shape spelling
/// is shared by SerializeOpParams and DumpOpParams without duplication.
///
/// @param target_shape Reshape target shape dimensions to serialize.
/// @param os Output stream to write to.
void SerializeReshapeShape(const std::vector<ReshapeDim>& target_shape, std::ostream& os);

/// @brief Serializes a Permute permutation to its canonical textual form.
///
/// Example: `[2,0,1]`. Tokens are unsigned decimals joined with commas and
/// no interior whitespace. An empty vector emits `[]` (rank zero). Exposed so
/// serde and dump share the canonical spelling without duplication.
///
/// @param permutation Axis permutation to serialize.
/// @param os Output stream to write to.
void SerializePermutation(const std::vector<uint32_t>& permutation, std::ostream& os);

} // namespace aethermind

#endif // AETHERMIND_OPERATORS_OP_PARAMS_SERDE_H
