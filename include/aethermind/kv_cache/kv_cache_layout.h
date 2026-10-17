#ifndef AETHERMIND_KV_CACHE_KV_CACHE_LAYOUT_H
#define AETHERMIND_KV_CACHE_KV_CACHE_LAYOUT_H

/// @file kv_cache_layout.h
/// @brief Physical geometry and byte-offset calculations for KV cache planes.

#include "aethermind/base/macros.h"
#include "aethermind/base/status.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>

namespace aethermind {

/// @brief Cache geometry and byte-offset computation for one key/value plane.
///
/// Each plane stores `num_layers * num_kv_heads * max_tokens * head_dim`
/// elements. `padded_head_dim` may exceed `head_dim` to allow alignment
/// padding between heads.
struct KVCacheLayout {
    size_t num_layers = 0;
    size_t num_kv_heads = 0;
    size_t max_tokens = 0;
    size_t head_dim = 0;
    /// Padded head width in elements; >= head_dim when alignment padding
    /// widens each token row.
    size_t padded_head_dim = 0;
    /// Byte offset between consecutive sequence positions.
    size_t token_stride = 0;
    /// Byte offset between consecutive KV heads.
    size_t head_stride = 0;
    /// Byte offset between consecutive layers.
    size_t layer_stride = 0;
    DataType kv_dtype{};
    /// Row-layout alignment in bytes. `token_stride` must be a multiple of it,
    /// so every token row starts aligned relative to its plane base. Views
    /// binding this layout additionally require CPU plane bases to be aligned
    /// to it, so kernels observe absolutely aligned token rows. The backing
    /// buffer's own declared alignment may be lower than the actual address
    /// alignment; only the base addresses are checked.
    size_t alignment = 64;

    /// @brief Returns the element size of the KV dtype.
    AM_NODISCARD size_t ElementBytes() const noexcept;
    /// @brief Validates the layout geometry and alignment.
    Status Validate() const noexcept;
    /// @brief Computes the byte offset of an element within the plane.
    ///
    /// @param layer_idx Layer index.
    /// @param kv_head_idx KV head index.
    /// @param seq_pos Sequence position.
    /// @param dim_idx Dimension index within the head.
    /// @return The element's byte offset, or an error if an index or stride
    ///         computation is out of range or overflows.
    AM_NODISCARD StatusOr<size_t> Offset(size_t layer_idx,
                                         size_t kv_head_idx,
                                         size_t seq_pos,
                                         size_t dim_idx) const noexcept;
    /// @brief Returns the total bytes of one key/value plane.
    AM_NODISCARD StatusOr<size_t> BytesPerPlane() const noexcept;

private:
    friend class KVCacheSessionView;

    // The caller must validate this layout in the current operation. These
    // helpers retain index and arithmetic checks; they cache no validation state.
    AM_NODISCARD StatusOr<size_t> OffsetValidated(size_t layer_idx,
                                                  size_t kv_head_idx,
                                                  size_t seq_pos,
                                                  size_t dim_idx) const;
    AM_NODISCARD StatusOr<size_t> BytesPerPlaneValidated() const;
};

} // namespace aethermind

#endif // AETHERMIND_KV_CACHE_KV_CACHE_LAYOUT_H
