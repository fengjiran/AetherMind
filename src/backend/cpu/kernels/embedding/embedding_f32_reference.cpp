#include "aethermind/base/check.h"
#include "aethermind/operators/ops/embedding_op.h"
#include "embedding_internal.h"

#include <algorithm>

namespace aethermind::cpu::detail {
namespace {

int64_t ReadTokenId(const void* token_ids_data,
                    const DataType& dtype,
                    size_t index) noexcept {
    // BuildEmbeddingF32ReferenceArgs validates dtype against the semantic contract first,
    // keeping the read path tied to kEmbeddingSupportedTokenIdDTypes.
    AM_DCHECK(IsSupportedTokenIdDType(dtype));
    if (dtype == DataType::Int(32)) {
        return *(static_cast<const int32_t*>(token_ids_data) + index);
    }

    if (dtype == DataType::UInt(32)) {
        return *(static_cast<const uint32_t*>(token_ids_data) + index);
    }

    if (dtype == DataType::Int(64)) {
        return *(static_cast<const int64_t*>(token_ids_data) + index);
    }
    // Unreachable after BuildEmbeddingF32ReferenceArgs validation; fall back without reading.
    return 0;
}

} // namespace

Status RunEmbeddingF32Reference(const EmbeddingF32KernelArgs& args) noexcept {
    if (args.token_count == 0) {
        return Status::Ok();
    }

    for (int64_t i = 0; i < args.token_count; ++i) {
        const int64_t token_id = ReadTokenId(args.token_ids_data,
                                             args.token_dtype,
                                             static_cast<size_t>(i));
        if (token_id < 0 || token_id >= args.vocab_size) {
            return Status::OutOfRange(
                    "EmbeddingKernel token id is out of vocabulary range");
        }

        const auto source_offset = static_cast<size_t>(token_id) *
                                   static_cast<size_t>(args.hidden_size);
        const auto target_offset = static_cast<size_t>(i) *
                                   static_cast<size_t>(args.hidden_size);
        std::copy_n(args.weight_data + source_offset, args.hidden_size,
                    args.output_data + target_offset);
    }

    return Status::Ok();
}

} // namespace aethermind::cpu::detail
