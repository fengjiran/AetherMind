#include "aethermind/inference/weight_binding_metadata.h"
#include "utils/overflow_check.h"

#include <new>
#include <utility>

namespace aethermind {
namespace {

StatusOr<std::vector<int64_t>> CompactRowMajorStrides(const std::vector<int64_t>& shape) {
    // Match TensorView's forward element-count calculation as well as the
    // reverse stride calculation below; a later zero must not hide overflow.
    uint64_t elements = 1;
    for (const auto extent: shape) {
        if (extent < 0) {
            return Status::InvalidArgument("Weight binding shape has a negative dimension");
        }
        if (CheckOverflowMul(elements, static_cast<uint64_t>(extent), &elements)) {
            return Status::Overflow("Weight binding shape product overflow");
        }
    }
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        // A zero total element count does not make preceding suffix products
        // representable; each stored stride must be checked independently.
        if (CheckOverflowMul(stride, shape[i], &stride)) {
            return Status::Overflow("Weight binding compact row-major stride calculation overflow");
        }
    }
    return strides;
}

} // namespace

StatusOr<TensorView> WeightBindingMetadata::Add(const void* data, DataType dtype,
                                                std::vector<int64_t> shape) try {
    AM_ASSIGN_OR_RETURN(auto strides, CompactRowMajorStrides(shape));
    // mmap and other byte-oriented sources do not guarantee alignment for typed
    // loads. Check the address before publishing a binding; SIMD alignment is
    // a separate kernel contract, not implied by the scalar lane type.
    const size_t alignment = dtype.natural_alignment();
    if (reinterpret_cast<uintptr_t>(data) % alignment != 0) {
        return Status::InvalidArgument("Weight binding data does not satisfy natural alignment");
    }
    entries_.push_back(Entry{.shape = std::move(shape), .strides = std::move(strides)});
    Entry& entry = entries_.back();
    return TensorView{data, dtype, IntArrayView{entry.shape},
                      IntArrayView{entry.strides}, alignment};
} catch (const std::bad_alloc&) {
    return Status::ResourceExhausted({});
}

size_t WeightBindingMetadata::size() const noexcept {
    return entries_.size();
}

bool WeightBindingMetadata::empty() const noexcept {
    return entries_.empty();
}

} // namespace aethermind
