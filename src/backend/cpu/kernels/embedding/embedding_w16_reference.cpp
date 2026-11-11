#include "embedding_internal.h"

#include <cstdint>
#include <cstring>

namespace aethermind::cpu::detail {
namespace {

int64_t ReadW16EmbeddingToken(const EmbeddingF32W16KernelArgs& args,
                              size_t index) noexcept {
    if (args.token_dtype == DataType::Int(32)) {
        int32_t token;
        std::memcpy(&token, args.token_ids + index * sizeof(token), sizeof(token));
        return token;
    }

    if (args.token_dtype == DataType::UInt(32)) {
        uint32_t token;
        std::memcpy(&token, args.token_ids + index * sizeof(token), sizeof(token));
        return token;
    }

    int64_t token;
    std::memcpy(&token, args.token_ids + index * sizeof(token), sizeof(token));
    return token;
}

} // namespace

Status RunEmbeddingF32W16Reference(const EmbeddingF32W16KernelArgs& args) noexcept {
    for (size_t index = 0; index < args.token_count; ++index) {
        const int64_t token = ReadW16EmbeddingToken(args, index);
        if (token < 0 || static_cast<uint64_t>(token) >= args.vocab_size) {
            return Status::OutOfRange("W16 Embedding token id is out of vocabulary range");
        }
        args.widen(args.weight + static_cast<size_t>(token) * args.weight_row_bytes,
                   args.output + index * args.output_row_bytes, args.hidden_size);
    }
    return Status::Ok();
}

} // namespace aethermind::cpu::detail
