#ifndef AETHERMIND_INFERENCE_WEIGHT_BINDING_STORAGE_H
#define AETHERMIND_INFERENCE_WEIGHT_BINDING_STORAGE_H

/// @file weight_binding_storage.h
/// @brief Stable shape/stride metadata for long-lived external tensor bindings.

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
/// prefill and again for decode. This storage owns one entry per bound value so
/// every view it issued stays readable for as long as the storage lives.
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
class WeightBindingStorage {
public:
    WeightBindingStorage() = default;
    WeightBindingStorage(WeightBindingStorage&&) noexcept = default;
    WeightBindingStorage& operator=(WeightBindingStorage&&) noexcept = default;
    WeightBindingStorage(const WeightBindingStorage&) = delete;
    WeightBindingStorage& operator=(const WeightBindingStorage&) = delete;
    ~WeightBindingStorage() = default;

    /// @brief Appends one entry for borrowed data and returns a view of it.
    ///
    /// Strides are derived as compact row-major. Alignment is left unspecified
    /// because the raw weight source carries none.
    ///
    /// @param data Borrowed element data; must outlive this storage.
    /// @param dtype Element type of `data`.
    /// @param shape Logical dimensions with non-negative extents, copied into
    ///        owned storage.
    /// @return View borrowing `data` and this storage's metadata.
    TensorView Add(const void* data, DataType dtype, std::vector<int64_t> shape);

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
