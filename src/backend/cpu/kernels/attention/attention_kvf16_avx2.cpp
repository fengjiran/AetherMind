#include "attention_decode_avx2_internal.h"
#include "attention_prefill_avx2_internal.h"

namespace aethermind::cpu::detail {

Status RunAttentionKVF16Avx2(const AttentionA32KernelArgs& args, const KVCacheReadBinding& read) noexcept {
    if (!HasVectorWidth(args)) {
        return RunAttentionKV16Reference(args, read, DataType::Float(16));
    }
    if (UseGroupedDecode(args, read)) {
        return RunGroupedDecode<KVElement::kF16>(args, read);
    }
    if (args.seq_len >= kMinPairedPrefillRows) {
        return RunPairedPrefill<KVElement::kF16>(args, read);
    }
    return RunContiguousHeads<KVElement::kF16>(args, read);
}

} // namespace aethermind::cpu::detail
