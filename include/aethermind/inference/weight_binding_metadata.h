#ifndef AETHERMIND_INFERENCE_WEIGHT_BINDING_METADATA_H
#define AETHERMIND_INFERENCE_WEIGHT_BINDING_METADATA_H

/// @file weight_binding_metadata.h
/// @brief Stable shape/stride metadata for long-lived external tensor bindings.

#include "aethermind/base/status.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/dtypes/data_type.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace aethermind {

/// @brief Owns the metadata arrays that immutable weight bindings borrow.
///
/// A TensorView borrows its shape and stride arrays, and an immutable binding
/// table is handed out repeatedly — once per PrepareExecutionBindings call, for
/// prefill and again for decode. This object owns one entry per bound value so
/// every view it issued stays readable for as long as it lives.
///
/// Stability contract:
/// - views issued earlier stay valid across later Add() calls, because growing
///   the entry vector move-constructs entries while each entry's inner
///   std::vector<int64_t> keeps its own heap buffer;
/// - views stay valid across a move of this object, because the entry vector's
///   buffer is transferred rather than copied;
/// - views die with this object, so it must outlive every binding table built
///   from it.
///
/// Entries must therefore never hold shape or strides inline (std::array or a
/// small-buffer vector): that would relocate the arrays and dangle every view
/// issued so far.
///
/// Only the metadata is owned here. Element data stays borrowed from the weight
/// backing storage, which must outlive this object.
///
/// Move-only. Not thread-safe: callers must serialize concurrent access.
class WeightBindingMetadata {
public:
    WeightBindingMetadata() = default;
    WeightBindingMetadata(WeightBindingMetadata&&) noexcept = default;
    WeightBindingMetadata& operator=(WeightBindingMetadata&&) noexcept = default;
    WeightBindingMetadata(const WeightBindingMetadata&) = delete;
    WeightBindingMetadata& operator=(const WeightBindingMetadata&) = delete;
    ~WeightBindingMetadata() = default;

    /// @brief Appends one entry for borrowed data and returns a view of it.
    ///
    /// Strides are derived as compact row-major. Alignment is left unspecified
    /// because the raw weight source carries none.
    ///
    /// @param data Borrowed element data; must outlive this object.
    /// @param dtype Element type of `data`.
    /// @param shape Logical dimensions copied into owned storage. Extents must
    ///        be non-negative and every suffix product must fit in int64_t,
    ///        including for empty tensors. Forward element-count products must
    ///        also fit in uint64_t, as required by TensorView.
    /// @return View borrowing `data` and this object's metadata arrays, or
    ///         InvalidArgument for negative extents, Overflow for unrepresentable
    ///         products, or empty ResourceExhausted on allocation failure.
    /// @note Failure leaves existing entries and issued views unchanged.
    AM_NODISCARD StatusOr<TensorView> Add(const void* data, DataType dtype,
                                          std::vector<int64_t> shape);

    AM_NODISCARD size_t size() const noexcept;
    AM_NODISCARD bool empty() const noexcept;

private:
    struct Entry {
        std::vector<int64_t> shape{};
        std::vector<int64_t> strides{};
    };

    std::vector<Entry> entries_{};
};

} // namespace aethermind

#endif
