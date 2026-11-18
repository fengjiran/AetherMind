#include "attention_decode_avx2_internal.h"
#include "attention_prefill_avx2_internal.h"

namespace aethermind::cpu::detail {

Status RunAttentionF32Avx2(const AttentionA32KernelArgs& args, const KVCacheReadBinding& read) noexcept {
    if (!HasVectorWidth(args)) {
        return RunAttentionF32Reference(args, read);
    }
    if (UseGroupedDecode(args, read)) {
        return RunGroupedDecode<KVElement::kF32>(args, read);
    }
    if (args.seq_len >= kMinPairedPrefillRows) {
        return RunPairedPrefill<KVElement::kF32>(args, read);
    }
    return RunContiguousHeads<KVElement::kF32>(args, read);
}

Status RunAttentionKVBF16Avx2(const AttentionA32KernelArgs& args, const KVCacheReadBinding& read) noexcept {
    if (!HasVectorWidth(args)) {
        return RunAttentionKV16Reference(args, read, DataType::BFloat(16));
    }
    if (UseGroupedDecode(args, read)) {
        return RunGroupedDecode<KVElement::kBF16>(args, read);
    }
    if (args.seq_len >= kMinPairedPrefillRows) {
        return RunPairedPrefill<KVElement::kBF16>(args, read);
    }
    return RunContiguousHeads<KVElement::kBF16>(args, read);
}

Status RunAttentionBF16Avx2(const AttentionA16KernelArgs& args, const KVCacheReadBinding& read) noexcept {
    if (!HasVectorWidth(args) || static_cast<size_t>(args.head_dim) > kMaxA16HeadDim) {
        return RunAttentionA16Reference(args, read);
    }
    if (UseGroupedDecode(args, read)) {
        return RunGroupedDecode<KVElement::kBF16, true>(args, read);
    }
    if (args.seq_len >= kMinPairedPrefillRows) {
        return RunPairedPrefill<KVElement::kBF16, true>(args, read);
    }
    return RunContiguousHeads<KVElement::kBF16, true>(args, read);
}

} // namespace aethermind::cpu::detail
