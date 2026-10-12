#include "aethermind/inference/weight_binding_metadata.h"
#include "utils/overflow_check.h"

#include <utility>

namespace aethermind {
namespace {

std::vector<int64_t> CompactRowMajorStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        AM_CHECK(!CheckOverflowMul(stride, shape[i], &stride),
                 "Compact row-major stride calculation overflow");
    }
    return strides;
}

} // namespace

TensorView WeightBindingMetadata::Add(const void* data, DataType dtype,
                                      std::vector<int64_t> shape) {
    entries_.push_back(Entry{.shape = std::move(shape)});
    Entry& entry = entries_.back();
    entry.strides = CompactRowMajorStrides(entry.shape);
    return {data, dtype, IntArrayView{entry.shape},
            IntArrayView{entry.strides}};
}

size_t WeightBindingMetadata::size() const noexcept {
    return entries_.size();
}

bool WeightBindingMetadata::empty() const noexcept {
    return entries_.empty();
}

} // namespace aethermind
