#include "aethermind/operators/ops/kvcache_update_op.h"
#include "aethermind/operators/operator_inference.h"

namespace aethermind::detail {

namespace {

// Reports the first unsupported dtype among k/v/cache and enforces that all
// four inputs share the same dtype. Mixed dtypes between activations and cache
// are rejected: no implicit conversion is declared.
Status ValidateKVCacheUpdateDTypes(std::span<const TensorSpec> inputs) {
    const TensorSpec& k = inputs[0];
    const TensorSpec& v = inputs[1];
    const TensorSpec& k_cache = inputs[2];
    const TensorSpec& v_cache = inputs[3];

    if (!IsKVCacheUpdateSupportedDType(k.dtype)) {
        return Status::InvalidArgument(
                MakeKVCacheUpdateUnsupportedDTypeMessage("KVCacheUpdate k"));
    }

    if (v.dtype != k.dtype) {
        return Status::InvalidArgument(
                "KVCacheUpdate v dtype must match k dtype");
    }

    if (k_cache.dtype != k.dtype) {
        return Status::InvalidArgument(
                "KVCacheUpdate k_cache_in dtype must match k dtype");
    }

    if (v_cache.dtype != k.dtype) {
        return Status::InvalidArgument(
                "KVCacheUpdate v_cache_in dtype must match k dtype");
    }
    return Status::Ok();
}

// Validates rank and shape contracts:
//   k, v             : rank 2, shape [seq_len, hidden]
//   k_cache, v_cache : rank 3, shape [kv_heads, cache_len, head_dim]
//   k.shape == v.shape; k_cache.shape == v_cache.shape
//   hidden == kv_heads * head_dim  (when statically provable)
//   seq_len > 0, hidden > 0, kv_heads > 0, head_dim > 0 (when static);
//   cache_len static-zero rejected; static seq_len > cache_len rejected.
//   The latter is the graph-level necessary part of the capacity contract
//   (seq_len <= cache_len); the runtime sufficient part
//   (current_pos + seq_len <= capacity) lives in KVCacheView::ValidateWrite
//   and is not expressible in the graph.
Status ValidateKVCacheUpdateShapes(std::span<const TensorSpec> inputs) {
    const SymbolicShape& k_shape = inputs[0].shape;
    const SymbolicShape& v_shape = inputs[1].shape;
    const SymbolicShape& k_cache_shape = inputs[2].shape;
    const SymbolicShape& v_cache_shape = inputs[3].shape;

    if (!HasRank(k_shape, 2)) {
        return Status::InvalidArgument(
                "KVCacheUpdate k must be rank 2 [seq_len, hidden]");
    }

    if (!HasRank(v_shape, 2)) {
        return Status::InvalidArgument(
                "KVCacheUpdate v must be rank 2 [seq_len, hidden]");
    }

    if (!HasRank(k_cache_shape, 3)) {
        return Status::InvalidArgument(
                "KVCacheUpdate k_cache_in must be rank 3 [kv_heads, cache_len, head_dim]");
    }

    if (!HasRank(v_cache_shape, 3)) {
        return Status::InvalidArgument(
                "KVCacheUpdate v_cache_in must be rank 3 [kv_heads, cache_len, head_dim]");
    }

    const ShapeSymbol& k_t = k_shape[0];
    const ShapeSymbol& k_last = k_shape[1];
    const ShapeSymbol& v_t = v_shape[0];
    const ShapeSymbol& v_last = v_shape[1];

    if (!AreProvablyEqual(k_t, v_t) || !AreProvablyEqual(k_last, v_last)) {
        return Status::InvalidArgument(
                "KVCacheUpdate k and v shapes must be equal");
    }

    if (k_t.IsStatic() && k_t.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "KVCacheUpdate k sequence length T must be positive when static");
    }

    if (k_last.IsStatic() && k_last.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "KVCacheUpdate k last dim must be positive when static");
    }

    const ShapeSymbol& kv_heads = k_cache_shape[0];
    const ShapeSymbol& cache_len = k_cache_shape[1];
    const ShapeSymbol& head_dim = k_cache_shape[2];

    for (size_t i = 0; i < 3; ++i) {
        if (!AreProvablyEqual(k_cache_shape[i], v_cache_shape[i])) {
            return Status::InvalidArgument(
                    "KVCacheUpdate k_cache_in and v_cache_in shapes must be equal");
        }
    }

    if (kv_heads.IsStatic() && kv_heads.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "KVCacheUpdate cache num_kv_heads must be positive when static");
    }

    if (head_dim.IsStatic() && head_dim.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "KVCacheUpdate cache head_dim must be positive when static");
    }

    if (cache_len.IsStatic() && cache_len.GetStaticValue() <= 0) {
        return Status::InvalidArgument(
                "KVCacheUpdate cache capacity must be positive when static");
    }

    if (k_t.IsStatic() && cache_len.IsStatic() &&
        k_t.GetStaticValue() > cache_len.GetStaticValue()) {
        return Status::InvalidArgument(
                "KVCacheUpdate k sequence length T must not exceed cache capacity "
                "when both are static");
    }

    if (kv_heads.IsStatic() && head_dim.IsStatic()) {
        const int64_t expected_last = kv_heads.GetStaticValue() * head_dim.GetStaticValue();
        if (k_last.IsStatic() && k_last.GetStaticValue() != expected_last) {
            return Status::InvalidArgument(
                    "KVCacheUpdate k last dim must equal num_kv_heads * head_dim");
        }
    }

    return Status::Ok();
}

} // namespace

StatusOr<InferenceResult> InferKVCacheUpdate(const OpParams& params,
                                             std::span<const TensorSpec> inputs) {
    if (!std::holds_alternative<KVCacheUpdateParams>(params)) {
        return Status::InvalidArgument(
                "KVCacheUpdate node requires KVCacheUpdateParams");
    }

    AM_RETURN_IF_ERROR(ValidateInferenceInputCount(OpType::kKVCacheUpdate, inputs));
    AM_RETURN_IF_ERROR(ValidateKVCacheUpdateDTypes(inputs));
    AM_RETURN_IF_ERROR(ValidateKVCacheUpdateShapes(inputs));

    // Cache-out follows cache-in verbatim (shape and dtype). The dtype is now
    // guaranteed to match k/v by ValidateKVCacheUpdateDTypes.
    return InferenceResult{
            .outputs = {inputs[2], inputs[3]},
    };
}

} // namespace aethermind::detail
