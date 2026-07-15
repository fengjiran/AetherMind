#include "aethermind/base/tensor.h"
#include "aethermind/base/shape_and_stride.h"
#include "aethermind/base/tensor_view.h"
#include "aethermind/memory/allocator.h"
#include "aethermind/utils/overflow_check.h"
#include "container/array_view.h"
#include "device.h"
#include "scalar.h"
#include "utils/logging.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace aethermind {

TensorView Tensor::view() const noexcept {
    return TensorView(*this);
}

MutableTensorView Tensor::mutable_view() noexcept {
    return MutableTensorView(*this);
}

size_t Tensor::logical_nbytes() const noexcept {
    const auto n = numel();
    AM_CHECK(n >= 0);

    size_t logical_nbytes = 0;
    bool overflow = CheckOverflowMul(static_cast<size_t>(n), itemsize(), &logical_nbytes);
    AM_CHECK(!overflow);
    return logical_nbytes;
}

size_t Tensor::max_touched_span_bytes() const noexcept {
    if (!is_initialized() || numel() == 0) {
        return 0;
    }

    const auto max_elem_offset = max_touched_element_offset();
    AM_CHECK(max_elem_offset >= 0);
    const auto touched_elems = static_cast<size_t>(max_elem_offset) + 1;
    const auto elem_size = itemsize();
    size_t span_bytes = 0;
    bool overflow = CheckOverflowMul(touched_elems, elem_size, &span_bytes);
    AM_CHECK(!overflow);
    return span_bytes;
}

bool Tensor::storage_range_is_valid() const noexcept {
    if (!is_initialized()) {
        return false;
    }

    const auto buffer_bytes = buffer_.nbytes();
    if (byte_offset_ > buffer_bytes) {
        return false;
    }

    const auto span = max_touched_span_bytes();
    if (span == 0) {
        return true;
    }

    size_t res = 0;
    if (CheckOverflowAdd(byte_offset_, span, &res)) {
        return false;
    }

    return res <= buffer_bytes;
}

Tensor Tensor::slice(int32_t dim_idx, int64_t start, int64_t end, int64_t step) const noexcept {
    AM_CHECK(is_initialized(), "Cannot slice uninitialized tensor.");
    AM_CHECK(step > 0, "slice currently supports step > 0 only.");

    const int32_t r = rank();
    AM_CHECK(r >= 1, "slice expects rank >= 1 tensor.");

    int32_t normalized_dim = dim_idx;
    if (normalized_dim < 0) {
        normalized_dim += r;
    }
    AM_CHECK(normalized_dim >= 0 && normalized_dim < r, "slice dim out of range.");

    const int64_t dim_size = dim(normalized_dim);
    AM_CHECK(dim_size >= 0, "slice expects non-negative dimension size.");

    auto normalize_bound = [dim_size](int64_t idx) noexcept {
        int64_t v = idx;
        if (v < 0) {
            v += dim_size;
        }

        if (v < 0) {
            v = 0;
        }
        if (v > dim_size) {
            v = dim_size;
        }
        return v;
    };

    const int64_t begin = normalize_bound(start);
    int64_t finish = normalize_bound(end);
    if (finish < begin) {
        finish = begin;
    }

    const int64_t slice_extent = finish - begin;
    int64_t out_dim_size = 0;
    if (slice_extent > 0) {
        AM_CHECK(!CheckOverflowAdd(slice_extent, step - 1, &out_dim_size),
                 "slice extent overflow.");
        out_dim_size /= step;
    }

    std::array<int64_t, ShapeAndStride::kMaxRank> out_shape{};
    std::array<int64_t, ShapeAndStride::kMaxRank> out_strides{};

    for (int32_t i = 0; i < r; ++i) {
        out_shape[static_cast<size_t>(i)] = dim(i);
        out_strides[static_cast<size_t>(i)] = stride(i);
    }

    out_shape[static_cast<size_t>(normalized_dim)] = out_dim_size;

    int64_t stepped_stride = 0;
    AM_CHECK(!CheckOverflowMul(stride(normalized_dim), step, &stepped_stride),
             "slice stride overflow.");
    AM_CHECK(stepped_stride >= 0, "slice stride must be non-negative.");
    out_strides[static_cast<size_t>(normalized_dim)] = stepped_stride;

    size_t new_byte_offset = byte_offset_;
    if (begin > 0) {
        int64_t elem_delta_i64 = 0;
        AM_CHECK(!CheckOverflowMul(begin, stride(normalized_dim), &elem_delta_i64),
                 "slice element offset overflow.");
        AM_CHECK(elem_delta_i64 >= 0, "slice expects non-negative element offset delta.");

        const auto elem_delta = static_cast<uint64_t>(elem_delta_i64);
        AM_CHECK(elem_delta <= std::numeric_limits<size_t>::max(),
                 "slice element offset exceeds size_t range.");

        size_t byte_delta = 0;
        AM_CHECK(!CheckOverflowMul(elem_delta, itemsize(), &byte_delta),
                 "slice byte offset overflow.");

        size_t updated_offset = 0;
        AM_CHECK(!CheckOverflowAdd(new_byte_offset, byte_delta, &updated_offset),
                 "slice byte offset overflow.");
        new_byte_offset = updated_offset;
    }

    ShapeAndStride sliced_shape_and_strides(
            IntArrayView{out_shape.data(), static_cast<size_t>(r)},
            IntArrayView{out_strides.data(), static_cast<size_t>(r)});

    return {buffer_, new_byte_offset, dtype_, sliced_shape_and_strides};
}

void Tensor::validate() const {
    AM_CHECK(is_initialized(), "Tensor buffer is not initialized.");
    AM_CHECK(shape_and_strides_.is_initialized(), "Tensor metadata is not initialized.");
    AM_CHECK(byte_offset_ <= buffer_.nbytes(), "Tensor byte_offset out of buffer range.");
    const auto r = shape_and_strides_.size();
    AM_CHECK(r >= 0 && r <= ShapeAndStride::kMaxRank, "Invalid tensor rank.");
    AM_CHECK(!dtype_.IsScalableVector(), "Tensor storage does not support scalable vector DataType.");
    AM_CHECK(itemsize() > 0, "Tensor dtype itemsize must be positive.");

    for (int i = 0; i < r; ++i) {
        AM_CHECK(shape_and_strides_.dim(i) >= 0, "Tensor shape must be non-negative.");
        AM_CHECK(shape_and_strides_.stride(i) >= 0, "Tensor stride requires non-negative.");
    }

    AM_CHECK(numel() >= 0);
    AM_CHECK(max_touched_element_offset() >= 0);
    AM_CHECK(storage_range_is_valid());
}

// ── Scalar-Tensor bridge helpers ────────────────────────────────
namespace {

void WriteScalarToBuffer(const DataType& dtype, const Scalar& scalar, void* data) {
#define WRITE_DISPATCH(code, bits, lanes, cpp_type, name)      \
    if (dtype == DataType(code, bits, lanes)) {                \
        *static_cast<cpp_type*>(data) = scalar.to<cpp_type>(); \
        return;                                                \
    }
    SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(WRITE_DISPATCH);
#undef WRITE_DISPATCH

    AM_THROW(RuntimeError) << "Unsupported dtype in WriteScalarToBuffer: " << dtype;
    AM_UNREACHABLE();
}

Scalar ReadScalarFromBuffer(const DataType& dtype, const void* data) {
#define READ_DISPATCH(code, bits, lanes, cpp_type, name)    \
    if (dtype == DataType(code, bits, lanes)) {             \
        return Scalar(*static_cast<const cpp_type*>(data)); \
    }
    SCALAR_TYPE_TO_CPP_TYPE_AND_NAME(READ_DISPATCH);
#undef READ_DISPATCH

    AM_THROW(RuntimeError) << "Unsupported dtype in ReadScalarFromBuffer: " << dtype;
    AM_UNREACHABLE();
}

}// namespace

Tensor Tensor::FromScalar(const Scalar& scalar, Allocator& allocator) {
    AM_CHECK(allocator.device().is_cpu(), "FromScalar requires a CPU allocator.");

    const auto dtype = scalar.type();
    const auto nbytes = static_cast<size_t>(dtype.nbytes());
    AM_CHECK(nbytes > 0, "FromScalar: dtype itemsize must be positive.");

    auto buffer = allocator.Allocate(nbytes);
    AM_CHECK(buffer.is_initialized(), "FromScalar: buffer allocation failed.");
    AM_CHECK(buffer.nbytes() >= nbytes, "FromScalar: allocated buffer too small.");

    void* data = buffer.mutable_data();
    AM_CHECK(data != nullptr, "FromScalar: buffer data pointer is null.");

    WriteScalarToBuffer(dtype, scalar, data);

    ShapeAndStride sas;
    sas.set_contiguous(IntArrayView{});

    return {std::move(buffer), 0, dtype, sas};
}

Scalar Tensor::item() const {
    AM_CHECK(is_initialized(), "item() requires an initialized Tensor.");
    AM_CHECK(device().is_cpu(), "item() requires a CPU Tensor.");
    AM_CHECK(numel() == 1, "item() requires numel() == 1.");
    AM_CHECK(data() != nullptr, "item() requires non-null data.");
    AM_CHECK(dtype_.IsScalar(), "item() requires a scalar dtype.");

    return ReadScalarFromBuffer(dtype_, data());
}

void Tensor::set_item(const Scalar& scalar) {
    AM_CHECK(is_initialized(), "set_item() requires an initialized Tensor.");
    AM_CHECK(device().is_cpu(), "set_item() requires a CPU Tensor.");
    AM_CHECK(numel() == 1, "set_item() requires numel() == 1.");
    AM_CHECK(mutable_data() != nullptr, "set_item() requires non-null data.");
    AM_CHECK(dtype_.IsScalar(), "set_item() requires a scalar dtype.");

    WriteScalarToBuffer(dtype_, scalar, mutable_data());
}

}// namespace aethermind
