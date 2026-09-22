#include "aethermind/inference/weight_binding_storage.h"

#include <utility>

namespace aethermind {
namespace {

std::vector<int64_t> CompactRowMajorStrides(const std::vector<int64_t>& shape) {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

} // namespace

TensorView WeightBindingStorage::Add(const void* data, DataType dtype,
                                     std::vector<int64_t> shape) {
    entries_.push_back(Entry{.shape = std::move(shape)});
    Entry& entry = entries_.back();
    entry.strides = CompactRowMajorStrides(entry.shape);
    return TensorView(data, dtype, IntArrayView{entry.shape}, IntArrayView{entry.strides});
}

size_t WeightBindingStorage::size() const noexcept {
    return entries_.size();
}

bool WeightBindingStorage::empty() const noexcept {
    return entries_.empty();
}

} // namespace aethermind
